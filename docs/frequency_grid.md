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

### Exact scores, no fitting

KL divergence from the reference (0 is perfect; about 1 or less is converged), and the number of modes,
for the automatic grid (`build_grid()`'s default, as shipped) and the fixed 1 per cent grid it replaced:

| campaign | automatic | fixed 1% |
|---|---|---|
| 6 Swift-like bands, 200 days, daily, SNR 100 | 377 modes, 0.02 | 327, 0.71 |
| the same, 100 days | 191, 0.11 | 272, 0.05 |
| the same, 650 days with seasonal gaps | 813, 0.04 | 426, 37 |
| the same, 650 days without gaps | 1010, 0.05 | 433, 133 |
| the same, 1500 days, every 2 days, gaps | 1258, 0.15 | 514, 214 |
| 6-hourly cadence, 100 days | 306, 0.07 | 273, 2.1 |
| 3-day cadence | 199, 0.43 | 327, 0.20 |
| SNR 30 | 215, 0.03 | 327, 0.02 |
| SNR 700 | 823, 0.02 | 327, 96 |
| black hole of 1e7 Msun | 446, 0.25 | 327, 16 |
| black hole of 2.5e8 Msun | 284, 0.04 | 327, 0.48 |
| g and i only | 159, 0.05 | 327, 0.00 |
| driver spectrum steepening to f^-3 | 238, 0.14 | 327, 0.33 |
| the same at SNR 700 | 434, 1.02 | 327, 187 |
| with an X-ray light curve of the driver | 1323, 0.05 | 327, 6.5 |
| Fairall 9's cadence and errors | 797, 0.04 | 428, 41 |
| NGC 5548's cadence and errors (19 light curves) | 768, 0.02 | 357, 4.3 |
| free lags, 200 days | 1825, 0.22 | 327, 68 |
| free lags, Fairall 9's cadence | 2494, 0.04 | 428, 35 |

The steep spectrum at SNR 700 is the one case near the limit: the estimate assumes a smoothed random
walk, and a steeper spectrum has more high-frequency signal than that at high signal-to-noise. It is why
the safety factor is 3 rather than 2 (KL 1.67 at 2). Driver light curves and free-lag bands are estimated with the model's own unsmoothed shape, so the
safety factor that guards against a wrong shape is 1 for them (`UNSMOOTHED_SAFETY`): with 3 they got 2 to
2.5 times the modes the fitted model needed, because tripling a w^-2 signal-to-noise pushes the harmonic
region sqrt(3) further out. At 1: X-ray driver 738 modes (was 1323, KL 0.04), free lags 1551 and 1248 (were
1825 and 2494; KL 0.19 and 0.30). Estimating them with smoothing instead left both free-lag cases unconverged
(KL about 5), so it is off. The table above was made with the factor at 3.

The post-fit check's prediction on these grids was 0.01 to 0.42 for the automatic grid apart from the
3-day cadence (1.45, KL 0.43) and the steep SNR-700 case (4.1, KL 1.02), and 1.9 to 361 on every
unconverged fixed grid. Hence its threshold of 2.

### Fits

Nested Laplace solves (`.optimise()`) on Kaggle (4 CPUs), against a fit on a much finer grid (0.25 per
cent log steps, or the rule with the true signal-to-noise). Shift: the largest of the accretion rate's and
the inclination's posterior-mean differences from the reference, in posterior standard deviations.

| campaign | automatic: modes, time, shift, ln Z | fixed 1%: modes, time, shift, ln Z |
|---|---|---|
| 6 bands, SNR 100 | 377, 483 s, 0.03, 0.0 | 327, 321 s, 0.10, −0.5 |
| 1e7 Msun | 446, 759 s, 0.04, −0.3 | 327, 421 s, 0.44, −11.7 |
| 3-day cadence | 199, 388 s, 0.02, −0.1 | 327, 325 s, 0.06, −1.1 |
| SNR 700 | 823, 2212 s, 0.07, +0.3 | 327, 394 s, 0.52, −125 |
| X-ray driver light curve | 1323, 3842 s, 0.17, +1.6 | 327, 422 s, 0.53, −8.1 |
| 650 days | 813, 1824 s, 0.10, +1.0 | 426, 467 s, 0.67, −64 |
| Fairall 9's cadence | 797, 1767 s, 0.08, +0.2 | 428, 535 s, 0.75, −28 |
| NGC 5548's cadence | 768, 3852 s, 0.04, −0.2 | 357, 873 s, 0.16, −5.0 |

Every automatic fit meets the agreed bar. The fixed grid missed it on six of eight: five by posterior
shifts of 0.39 to 0.75 standard deviations (the accretion rate always higher), and NGC 5548's by its KL
(4.3). The post-fit check flagged each of those six and, at its threshold of 2, none of the automatic fits (the
highest was the 3-day cadence's 1.5). The price is time on rich data: 2 to 9 times longer, which is what a converged fit of those
campaigns costs. On sparse or noisy data the automatic grid is smaller than the fixed one.
