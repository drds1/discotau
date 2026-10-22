# Faster direct solves (design, in progress)

This page is the working design for making `EchoFit.optimise()` (the nested Laplace solve and the
single-Gaussian solve) much faster. Nothing on it is in the library yet. It records the structure being
exploited, the maths, what has been verified and the plan.

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

## Verified so far

On Fairall 9's six Swift light curves (3091 points, 428 modes, p = 874), at the cached optimum of the
face-on disc fit, on a laptop:

| | log likelihood | time |
|---|---|---|
| dense float64 covariance (reference) | 602.845803 | 0.8 s |
| Gram formulation, value | 602.845803 (difference 5e-9) | 51 ms |
| Gram formulation, value and gradient | gradient matches finite differences to 3e-5 | 157 ms |
| inner loop at one outer point, value and gradient in gamma | difference 4e-11 | 64 ms (plus 78 ms once per point) |
| discotau today, jitted potential | | 328 ms |
| discotau today, jitted potential and gradient | | 459 ms |

Kaggle benchmarks at the larger grids (time and peak memory, current against Gram) are running.

## Plan

1. Benchmark at scale on Kaggle (time and memory at 300 to 1150 modes).
2. Replace `model._marginal_solve` with the Gram likelihood (float64 via `pure_callback`, `custom_vjp`).
   `reverberation_model` passes each band's transfer coefficients and its fixed basis instead of the
   combined columns. All three solvers and NUTS keep working unchanged. The error model uses the low-rank
   weight family.
3. Restructure the nested solve's inner loop around the gamma scaling (responses once per outer point,
   inner Newton on a few scalars).
4. Check posteriors and evidence against the current solves on the CREAM-paper cases, Fairall 9 and
   NGC 5548, and time them.

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
