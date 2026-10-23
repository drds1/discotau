# Faster direct solves

How `EchoFit.optimise()` (the nested Laplace solve and the single-Gaussian solve) and NUTS were made
faster: the structure exploited, the maths, what was measured, and what was tried and dropped.

## Why the solves are slow

Every direct solve evaluates the marginal likelihood of the light curves, with the driver's Fourier
coefficients, offsets and backgrounds integrated out exactly,

```
y ~ N(m, D + M M^T),
```

and its gradient, thousands of times. The nested solve needs about 2400 gradients plus its inner Hessians.
Each evaluation today

- recomputes every band's response and its transfer coefficients, and
- factorises the stacked `(N + p) x p` system `[D^-1/2 M; I]` by QR (decision #21 in `CLAUDE.md`),

so it costs `O((N + p) p^2)`, with N the number of data points and `p = 2 x modes + offsets +
backgrounds`. Reverse-mode differentiation keeps the QR's factors, and the nested solve's `vmap` over grid
points multiplies that memory by the batch size. On Kaggle (4 vCPU, 31 GB), Fairall 9- and NGC 5548-sized
problems with 900 to 1150 driver modes ran out of memory, and a 1056-mode Fairall 9-cadence fit took four
hours.

The frequency-grid study (October 2026) found that long or high signal-to-noise campaigns genuinely need
grids of that size, so the cost has to come down rather than the grid.

## The structure being wasted

For band (or driver light curve) `b`, the design matrix factorises as `M_b = Phi_b R_b`.

- `Phi_b = [sin(w t), cos(w t)]` at the band's observation times **never changes during a fit**.
- `R_b` is block-diagonal in the modes, one 2 x 2 rotation and scale per mode,
  `[[c A, c B], [-c B, c A]]` with `c = gain_b sigma_drw s_k` and `(A, B)` the response's transfer
  coefficients. **Only `R_b` depends on the nonlinear parameters.**

So the Gram matrix `G_b = Phi_b^T D_b^-1 Phi_b`, the projection `h_b = Phi_b^T D_b^-1 y_b` and the offset
and background equivalents can all be computed once per fit. Then

```
P = I + sum_b R_b^T G_b R_b
log p(y) = -1/2 [y^T D^-1 y - b^T P^-1 b + log det P + log det D + N log 2 pi]
```

costs `O(B p^2 + p^3)` per evaluation, independent of N, and no `N x p` matrix is ever formed.

### The gradient

With `u = P^-1 b` and `W = P^-1 + u u^T`, the exact gradient with respect to each band's `R_b` is

```
dL/dR_b = h_b u^T - G_b R_b W - G_E,b W_E.
```

Only the diagonals of its 2 x 2 sub-blocks are needed (`O(K^2)` per band, K modes). Everything upstream of
`(c A, c B)`, namely the responses, gains and the prior's power spectrum, stays in JAX and reaches the
likelihood through a `custom_vjp`.

### Precision

The textbook quadratic form `y^T D^-1 y - b^T P^-1 b` cancels catastrophically in float32, which is why
decision #21 moved to QR. Evaluated on the host in float64 (LAPACK through `jax.pure_callback`), it is exact,
and nothing switches on JAX's process-wide `jax_enable_x64`. Float64 gradients also make the
finite-difference Hessians the solves rely on much cleaner.

### Building the Gram matrices

`G_b`'s entries are the spectral window of the band's sampling, `sum_i e^(i nu t_i) / sigma_i^2`, at sums
and differences of grid frequencies. It is Toeplitz on the harmonic part of the grid. Direct construction
costs `O(N K^2)` once per fit. Non-uniform FFTs (Press & Rybicki 1989, the extirpolation behind fast
Lomb-Scargle; nifty-ls, arXiv:2409.08090) would bring it to `O(N log N + K^2)` if that ever matters.

## The error model as a low-rank family of weights

With the error model on (`fit_error_model=True`, needed for real multi-band data),
`sigma_i^2 = s^2 e_i^2 + j^2`, so

```
D^-1 = (1/s^2) diag(1 / (e_i^2 + u)),   u = j^2 / s^2.
```

The family `u -> 1/(e_i^2 + u)` has low numerical rank over one band's quoted errors. On Fairall 9's six
Swift bands, over seven decades of `u`, 10 to 12 components reach 1e-9 relative accuracy. This is the
data-adaptive relative of the exponential-sum approximations of 1/x (Beylkin & Monzon 2005, 2010; Braess &
Hackbusch 2005; Hackbusch 2019). With an orthonormal basis `U_b` of that family, any `u` gives exact
coefficients `c(u) = U_b^T (1/(e^2 + u))` in `O(N_b r)`, and

```
G_b(s, j) = (1/s^2) sum_r c_r(u) G_b^(r),   G_b^(r) = Phi_b^T diag(U_b[:, r]) Phi_b   (precomputed).
```

The error model then costs `r` times the assembly instead of a rebuild.

## The nested solve's inner loop

At one outer grid point (the accretion rate, the inclination and the temperature slope if fitted) the
responses are fixed, and the inner parameters enter only as scalars. With `gamma_b = gain_b sigma_drw`,
`R_b = gamma_b R_hat_b`, so

```
P(gamma, s, u) = I + sum_b (gamma_b^2 / s_b^2) sum_r c_r(u_b) H_b^(r)   (plus the offset and background rows)
```

with `H_b^(r) = R_hat_b^T G_b^(r) R_hat_b` built once per outer point. Each inner evaluation is then a
weighted sum of fixed matrices and one Cholesky, with an analytic gradient, and the responses are never
recomputed inside the inner loop. Today every inner gradient recomputes all responses, all transfer
coefficients and the full QR.

## What is in the library (branch `fast-direct-solves`)

1. **Float64 marginal likelihood on the host** (`discotau/gram.py`, `GramCache`), through
   `jax.pure_callback` with an analytic `custom_vjp`. Each light curve enters P the faster way for it
   (`HOST_METHOD = "auto"`): light curves whose noise is fixed through precomputed Gram matrices, light
   curves with the error model through their rows of the weighted design matrix and one BLAS-3 `syrk`
   (compute-bound, the error model exact). `model.MARGINAL_METHOD = "auto"` keeps the QR when there are
   fewer than 1.2 data points per linear parameter, where its data-space form is cheaper.
2. **Exact Hessian for the scalar parameters** (`GramCache.scalar_hessian`, `discotau/exact_hessian.py`):
   every light curve's log scale (gain times sigma_drw) and error scale and jitter, from the Gaussian
   Hessian identity moved into p x p quantities. One T1 + T2 pair cancels to ~1e-3 of its parts and is
   combined analytically. Used by both direct solves when the Gram path is on; other parameters keep
   finite-difference columns.
3. **Hessians reused** across Newton steps and between restarts that land together.
4. **Draw scoring in memory-bounded chunks** (the cause of the out-of-memory kills).
5. **`fit(init_from_optimum=True, mass_from_optimum=True)`**: NUTS starts with the direct solve's
   covariance as its inverse mass matrix.

## Measured on Kaggle (simulated light curves with known truths, `experiments/fast_solve/`)

Answers are unchanged throughout: log evidence within ~0.2 for the nested solve, posteriors within 0.01
dex. Whole solves, seconds, with the error model on every light curve:

| solve | scenario | before | after | speed-up |
|---|---|---|---|---|
| single Gaussian | 6 bands, 1216 points | 766 | 305 | 2.5x |
| single Gaussian | 6 bands + X-ray driver | 701 | 272 | 2.6x |
| single Gaussian | NGC 5548 cadence, 19 light curves | 1731 | 627 | 2.8x |
| nested | 6 bands | 1883 | 884 | 2.1x |
| nested | 6 bands + X-ray driver | 2725 | 1252 | 2.2x |
| nested | NGC 5548 cadence, 19 light curves | 6438 | 2302 | 2.8x |

Without the error model, on long well-sampled campaigns, the likelihood alone is 2.5-3x faster per
gradient (Fairall 9's cadence 0.66 s to 0.23 s) and whole nested solves ~2x.

NUTS (6 bands, 327 modes): the default sampled model (~670 dimensions) failed to mix (7.8 h, effective
sample size 7, stuck far from the truth); starting it at the optimum, with or without the solve's mass
matrix, still mixed badly (ESS 37-92 in 3.5 h). The marginalised model with the warm start was best:
1771 s against 5004 s with defaults, 0.233 against 0.159 effective samples per second, no divergences.

## Tried and dropped

- **Responses computed once per outer grid point** in the nested solve's inner loop: no gain with the
  error model (the host likelihood dominates each gradient), reverted.
- **In-place Gram assembly alone**: the host was memory-bound on many light curves; the dense `syrk`
  path replaced it for light curves with the error model.

## Literature

- Fourier-feature Gaussian processes, with hyperparameter-independent feature matrices precomputed once:
  variational Fourier features (Hensman, Durrande & Solin 2017), integrated variational Fourier features
  (arXiv:2308.14142), convolved multi-output processes through Fourier features (arXiv:1805.07460). Here
  the representation is exact rather than an approximation.
- Fast trigonometric sums for irregular sampling: Press & Rybicki (1989, ApJ 338, 277), nifty-ls
  (arXiv:2409.08090).
- Exponential sums for 1/x: Beylkin & Monzon (2005, 2010), Braess & Hackbusch (2005), Hackbusch (2019).
- INLA's cost is the log determinant and the linear solve: Rue, Martino & Chopin (2009),
  Gaedke-Merzhaeuser et al. (2023, arXiv:2303.15254); without sparsity, Wood (2020, Biometrika 107, 223).
