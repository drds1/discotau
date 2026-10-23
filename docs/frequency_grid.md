# The automatic driver frequency grid

How `build_grid()` chooses the driver's Fourier frequencies (the default since October 2026), why, and
how it was tested. Code: `discotau/frequency_grid.py`. The study behind it is in
`experiments/frequency_grid/` (untracked).

## Why the grid matters

The driver is a sum of sinusoids, $X(t) = \sum_k [S_k \sin \omega_k t + C_k \cos \omega_k t]$, with a
random-walk prior on the coefficients. With every nonlinear parameter fixed, the model is Gaussian in the
data, $y \sim N(m, D + M M^T)$, and the grid enters only through $M M^T$: a Riemann sum over frequency of
the light curves' cross-covariance. So a grid is good enough when that sum is accurate to within the noise.

A campaign of length $T$ resolves about one pair of Fourier coefficients per $2\pi/T$ of bandwidth. Let
$\rho(\omega)$ be the signal-to-noise per resolved pair, summed over light curves (they all see one
driver):

```math
\rho(\omega) = P(\omega)\,\frac{2\pi}{T} \sum_b g_b^2\, |\Psi_b(\omega)|^2 \sum_i \frac{1}{2\sigma_{bi}^2},
```

with $P$ the driver's power per unit $\omega$, $g_b$ band $b$'s gain and $\Psi_b$ its response. A grid
mode spaced $\Delta\omega$ from its neighbour stands for $m = \Delta\omega\, T / 2\pi$ resolved pairs. For
$m > 1$ it represents one of them and misses the rest, and the expected log evidence lost per resolved
pair is

```math
\ell(\rho, m) = \Big(1 - \frac{1}{m}\Big)\big(\rho - \ln(1 + \rho)\big)
 + \frac{1}{m}\Big(\frac{1 + \rho}{1 + m\rho} - 1 + \ln\frac{1 + m\rho}{1 + \rho}\Big),
```

zero for $m \le 1$. (The first term is the Kullback-Leibler divergence between a rank-one signal plus
noise and noise alone, for each missed pair; the second is the one mode carrying $m$ times the variance
it should.) The total loss is $\int \ell\, (T/2\pi)\, d\omega$. With the fitted model's own $\rho$ it
predicted 1 to 4 times the exact KL, on the safe side.

Two consequences:

- **Harmonic spacing is needed wherever $\rho \gtrsim 1$**, and the grid may thin where $\rho$ is small.
  For a given loss the fewest modes come from spacing proportional to $1/\rho$ (a Lagrangian argument in
  the small-$\rho$ limit, where $\ell \approx (m - 1)\rho^2/2$).
- **A coarse grid biases the fit, not just its evidence.** The loss depends on the parameters: narrow
  responses keep $\rho$ high to higher frequency. On the NGC 5548 cadence the 1 per cent grid lost 58 at
  the finest grid's optimum but only 3 at its own, which sat 0.2 dex higher in accretion rate. A coarse
  grid favours broad responses.

## The rule

1. **Estimate $\rho$ from the data alone.** Each light curve's power spectrum is fitted as a random walk
   smoothed on a timescale $b$ (the response's width), plus white noise beyond the quoted errors:
   $P_b(\omega) = A / ((\omega^2 + \omega_0^2)(1 + \omega^2 b^2))$, $\omega_0 = 1/2T$. Its
   autocovariance is a difference of two exponentials, so the Gaussian-process likelihood is exact; it is
   maximised by L-BFGS with the analytic gradient after a linear detrend (seconds per light curve; light
   curves longer than 1000 points are thinned for this step only). Driver light curves and free-lag bands
   are fitted with $b = 0$, as the model treats them. Periodograms do not work for this: irregular sampling
   leaks power and leaves a floor 30 to 100 times the noise.
2. **Multiply by 3** (`grid_safety`). The estimate assumes a smoothed random walk. A steeper driver
   spectrum at high signal-to-noise has more high-frequency signal than that, and 3 was needed there.
3. **Design the grid.** Start at $2\pi/P_\text{max}$ with its first 16 harmonics (slow trends). Above
   them, space modes by $(2\pi/T)\,\kappa/\rho$, never closer than $2\pi/(T + \tau_\text{max})$ (the
   window's own resolution) and never sparser than 3 per cent of $\omega$, with $\kappa$ the largest that
   loses at most 0.25. Stop where everything above loses at most 0.25, but not below $f_\text{max}$ and
   not beyond $4 f_\text{max}$.

**Why the window's resolution, not $2\pi/P_\text{max}$.** $P_\text{max} = 2(T + \tau_\text{max})$ lets the
series represent trends longer than the window. Spacing every mode at its harmonics, as the fixed grid
does below its crossover, puts about $2.4$ modes in each resolved pair, which costs time and buys nothing.
Spacing at $2\pi/(T + \tau_\text{max})$ cut the designed grids to a median 61 per cent of their size and
raised the worst KL from 0.40 to 0.43. Spacing at $2\pi/T$ went too far (KL up to 1.6).

## The check after a fit

`check_frequency_grid()` runs at the end of `.optimise()` and `.fit()`. It traces the fitted model once to
get each light curve's response, gain and effective noise, computes the model's own $\rho$, and predicts
the grid's loss. Above 2 it warns (the prediction ran 1 to 4 times the exact KL, on the safe side), and stores the grid the fitted model itself calls for in
`ef.grid_check["suggested_freqs"]`. It cannot see structure the model doesn't expect: real light curves
carried more high-frequency variability than their fitted models predicted (Fairall 9 and NGC 5548), so
the realised loss of a grid that stops early can exceed the prediction.

## How it was tested

19 simulated campaigns (`experiments/frequency_grid/sims.py`): red-noise drivers drawn on a much longer,
finer time grid (so not periodic on any fitting grid), every band the driver convolved with the
irradiated thin-disc response, real cadences for Fairall 9 and NGC 5548. Each grid was scored exactly, with
no fitting, by the KL divergence from all harmonics of $P_\text{max}$ to 4 cycles/day at the true
parameters. Agreed criterion for "converged": KL about 1 or less and posterior shifts under 0.2 standard
deviations.
