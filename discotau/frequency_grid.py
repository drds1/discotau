"""
The automatic driver frequency grid: as few modes as the data need.

Why a grid costs anything (``docs/frequency_grid.md``, CLAUDE.md decision #38):
a campaign of length T resolves one pair of Fourier coefficients per
2 pi / T of bandwidth. Let rho(w) be the signal-to-noise per resolved pair,
summed over light curves (they all see one driver):

    rho(w) = P(w) (2 pi / T) sum_b g_b^2 |Psi_b(w)|^2 sum_i 1 / (2 sigma_bi^2),

with P the driver's power per unit w and Psi_b band b's response. A grid
mode standing for m > 1 resolved pairs represents one of them and misses
m - 1, and the expected log evidence lost per resolved pair is

    (1 - 1/m)(rho - ln(1 + rho)) + (1/m)((1 + rho)/(1 + m rho) - 1 + ln((1 + m rho)/(1 + rho))),

zero for m <= 1 (:func:`loss_density`). So the grid needs fine spacing where
rho is large and may thin where it is small. For a given loss the fewest
modes come from spacing proportional to 1 / rho (:func:`design_grid`).

Before a fit, rho is estimated from each light curve alone, with an exact
Gaussian-process fit of a smoothed random walk plus excess white noise
(:func:`fit_light_curve_power`), and multiplied by a safety factor. After a
fit, :meth:`discotau.EchoFit.check_frequency_grid` recomputes rho from the
fitted model and warns if the grid is predicted to lose more than 2 in log
evidence (the prediction is conservative).

Tested exactly (KL against all harmonics to 4 cycles/day, no fitting) on 19
simulated campaigns (``experiments/frequency_grid/``): worst predicted loss
1.03, every other case 0.43 or less.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# Multiplies the estimated rho: the estimate assumes a smoothed random walk,
# and a steeper driver spectrum at high SNR has more high-frequency signal
# than that (the one simulated case near the limit, predicted loss 1.03).
SAFETY = 3.0
# Expected log-evidence loss allowed: half to thinning the spacing, half to
# stopping the grid early.
LOSS_BUDGET = 0.5
# The densest spacing is 2 pi / (T + tau_max), the window the driver must
# cover, after this many modes at 2 pi / period_max (period_max is twice that
# window, for slow trends). Spacing all modes at 2 pi / period_max needed ~1.6x
# the modes for the same loss.
N_TREND_MODES = 16
# The spacing above the harmonics is never sparser than this fraction of w.
MAX_LOG_STEP = 0.03
# The grid reaches at least f_max (cycles/day) and never extends past
# F_CAP_FACTOR * f_max, however large rho is there.
F_CAP_FACTOR = 4.0
# Light curves with fewer points are left out of the estimate (with a warning).
MIN_POINTS = 10
# Light curves longer than this are thinned evenly for the estimate (the noise
# term still uses every point).
MAX_POINTS = 1000
# Warn above this many modes: each gradient costs roughly O(modes^2 to modes^3).
LARGE_GRID_MODES = 3000
# The post-fit check warns above this predicted loss. Its prediction ran 1-4x
# the exact KL on the simulated campaigns (on the safe side), and a converged
# grid is KL ~1 or less: at 1 it warned falsely on a 3-day cadence (predicted
# 1.45, exact 0.43). Every unconverged grid tested predicted 1.9 or more.
CHECK_LOSS = 2.0
# Whether the estimate fits a smoothing timescale to driver light curves and
# free-lag bands too. False: they are estimated as unsmoothed random walks,
# as the model treats them, which over-provisions (2-4x the modes needed on
# the simulated campaigns). True halved an X-ray driver's grid at no cost but
# left both free-lag campaigns unconverged (KL ~5), so it stays off.
SMOOTH_ALL = False

TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------- the theory
def loss_density(r, m):
    """Expected log-evidence loss per resolved Fourier pair for summed
    signal-to-noise ``r`` and ``m`` resolved pairs per grid mode."""
    m = np.maximum(m, 1.0)
    miss = (1 - 1 / m) * (r - np.log1p(r))
    keep = (1 / m) * ((1 + r) / (1 + m * r) - 1 + np.log((1 + m * r) / (1 + r)))
    return miss + keep


def predicted_loss(rho_fn: Callable, grid, T: float, f_top: float) -> float:
    """Expected log-evidence loss of ``grid`` (angular frequencies) under
    ``rho_fn``, counting everything above the grid's top up to ``f_top``
    (cycles/day) as missed."""
    grid = np.sort(np.asarray(grid, dtype=float))
    w = np.geomspace(grid[0], max(2 * np.pi * f_top, grid[-1] * 1.0001), 6000)
    dw = np.interp(w, grid, np.gradient(grid) if len(grid) > 1 else np.full(1, grid[0]))
    m = dw * T / (2 * np.pi)
    r = np.maximum(rho_fn(w), 0.0)
    dens = np.where(w > grid[-1], r - np.log1p(r), loss_density(r, m))
    return float(TRAP(dens * T / (2 * np.pi), w))


def design_grid(rho_fn: Callable, T: float, period_max: float, window: float, f_min: float, f_cap: float,
                loss_budget: float = LOSS_BUDGET, n_trend: int = N_TREND_MODES,
                max_log_step: float = MAX_LOG_STEP, n_mesh: int = 4000) -> Tuple[np.ndarray, dict]:
    """Angular frequencies for the summed signal-to-noise ``rho_fn(w)``.

    Starts at ``2 pi / period_max`` with ``n_trend`` harmonics of it; above
    them the spacing is ``2 pi / T * kappa / rho(w)``, kept between
    ``2 pi / window`` and ``max_log_step * w``, with ``kappa`` the largest
    that loses at most half of ``loss_budget``. The grid stops where
    everything above loses the other half, but reaches at least ``f_min`` and
    at most ``f_cap`` (cycles/day)."""
    h = 2 * np.pi / period_max
    dw_floor = max(h, 2 * np.pi / window)
    budget_spacing = budget_tail = 0.5 * loss_budget
    w_cap = 2 * np.pi * f_cap
    w_mesh = np.geomspace(h, w_cap, n_mesh)
    r = np.maximum(rho_fn(w_mesh), 0.0)
    dens_tail = (T / (2 * np.pi)) * (r - np.log1p(r))
    seg = 0.5 * (dens_tail[1:] + dens_tail[:-1]) * np.diff(w_mesh)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    i_max = int(np.argmax(tail <= budget_tail))
    w_max = min(max(w_mesh[i_max], 2 * np.pi * f_min, (n_trend + 1) * h), w_cap)
    w_mesh = np.geomspace(h, w_max, n_mesh)
    r = np.maximum(rho_fn(w_mesh), 0.0)

    def spacing(kappa, w):
        w = np.atleast_1d(w)
        rr = np.maximum(rho_fn(w), 1e-300)
        return np.maximum(dw_floor, np.minimum((2 * np.pi / T) * kappa / rr, max_log_step * w))

    def loss(kappa):
        m = spacing(kappa, w_mesh) * T / (2 * np.pi)
        return float(TRAP(loss_density(r, m) * T / (2 * np.pi), w_mesh))

    lo, hi = 1e-6, 1e6
    for _ in range(80):  # bisect in log kappa: the largest kappa within budget
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if loss(mid) <= budget_spacing else (lo, mid)
    kappa = lo
    w = [h * (k + 1) for k in range(max(n_trend, 1))]
    while True:
        nxt = w[-1] + float(spacing(kappa, w[-1])[0])
        if nxt > w_max * (1 + 1e-9):
            break
        w.append(nxt)
    return np.asarray(w), dict(kappa=kappa, f_max=w_max / (2 * np.pi),
                               predicted_loss=loss(kappa) + float(tail[min(i_max, len(tail) - 1)]))


# ---------------------------------------------------------------- the estimate
def _gp_nll_grad(theta, D, r, yerr2, tau_long, smoothing):
    """Negative log likelihood of :func:`fit_light_curve_power`'s model and its
    gradient with respect to (ln A, ln b, ln c), or (ln A, ln c) without
    smoothing. With alpha = K^-1 r, d NLL = 1/2 sum((K^-1 - alpha alpha^T) * dK)."""
    import scipy.linalg as sla

    a = tau_long
    if smoothing:
        A, b, c = np.exp(theta)
    else:
        (A, c), b = np.exp(theta), 0.0
    Dabs = np.abs(D)
    ea = np.exp(-Dabs / a)
    if b > 1e-6:
        c0 = (np.pi / 2) * A * a ** 2 / (a ** 2 - b ** 2)
        eb = np.exp(-Dabs / b)
        Ks = c0 * (a * ea - b * eb)
    else:
        Ks = (np.pi / 2) * A * a * ea
    K = Ks.copy()
    K[np.diag_indices_from(K)] += yerr2 + c
    try:
        cf = sla.cho_factor(K, lower=True, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return 1e30, np.zeros_like(theta)
    alpha = sla.cho_solve(cf, r, check_finite=False)
    nll = 0.5 * float(r @ alpha) + float(np.log(np.diag(cf[0])).sum())
    if not np.isfinite(nll):
        return 1e30, np.zeros_like(theta)
    Q = sla.cho_solve(cf, np.eye(len(r)), check_finite=False) - np.outer(alpha, alpha)
    g_A = 0.5 * float(np.sum(Q * Ks))      # dK/dlnA = Ks
    g_c = 0.5 * float(np.trace(Q)) * c     # dK/dlnc = c I
    if not smoothing:
        return nll, np.array([g_A, g_c])
    dK_db = (2 * b / (a ** 2 - b ** 2)) * Ks - c0 * eb * (1.0 + Dabs / b)
    return nll, np.array([g_A, 0.5 * float(np.sum(Q * dK_db)) * b, g_c])


def fit_light_curve_power(t, y, yerr, smoothing: bool = True, max_points: int = MAX_POINTS) -> dict:
    """Maximum-likelihood power spectrum of one light curve: a random walk of
    power ``A / w^2`` above ``1 / (2 T)``, smoothed on a timescale ``b``
    (power ``A / ((w^2 + 1/(2T)^2)(1 + w^2 b^2))``), plus excess white noise of
    variance ``c`` on top of the quoted errors. Exact Gaussian likelihood
    (its autocovariance is a difference of two exponentials) after a weighted
    linear detrend, by L-BFGS with the analytic gradient. ``smoothing=False``
    fixes ``b = 0``. Returns ``A``, ``b``, ``c``, ``tau_long`` and ``nll``."""
    from scipy.optimize import minimize

    t, y, yerr = (np.asarray(a, dtype=np.float64) for a in (t, y, yerr))
    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]
    if len(t) > max_points:
        keep = np.linspace(0, len(t) - 1, max_points).astype(int)
        t, y, yerr = t[keep], y[keep], yerr[keep]
    T = max(float(np.ptp(t)), 1e-9)
    tau_long = 2.0 * T
    x = (t - t.mean()) / T
    coef = np.polyfit(x, y, 1, w=1.0 / yerr)
    r = y - np.polyval(coef, x)
    D = t[:, None] - t[None, :]
    yerr2 = yerr ** 2
    e2 = float(np.mean(yerr2))
    A0 = max(float(np.var(r)), 1e-30) / ((np.pi / 2) * tau_long)
    c0 = 0.1 * e2
    if smoothing:
        starts = [[np.log(A0), np.log(b0), np.log(c0)] for b0 in (0.3, 3.0)]
        bounds = [(None, None), (np.log(1e-3), np.log(0.5 * tau_long)), (np.log(1e-12 * e2), None)]
    else:
        starts = [[np.log(A0), np.log(c0)]]
        bounds = [(None, None), (np.log(1e-12 * e2), None)]
    best = None
    for th in starts:
        res = minimize(_gp_nll_grad, th, args=(D, r, yerr2, tau_long, smoothing), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=200, ftol=1e-10, gtol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    p = np.exp(best.x)
    return dict(A=float(p[0]), b=float(p[1]) if smoothing else 0.0, c=float(p[-1]), tau_long=tau_long,
                nll=float(best.fun))


def estimate_rho(curves: Dict[str, dict], T: float) -> Tuple[Optional[Callable], dict]:
    """Summed signal-to-noise per resolved pair, estimated from each light
    curve alone. ``curves``: name -> dict with ``t``, ``y``, ``yerr`` and
    ``smoothing`` (False for a light curve the model treats as the
    unsmoothed driver: a driver light curve or a free-lag band). Returns
    ``(rho_fn, fits)``, ``rho_fn`` None if no light curve could be used."""
    terms, fits = [], {}
    for name, d in curves.items():
        t, y, yerr = (np.asarray(d[k], dtype=np.float64) for k in ("t", "y", "yerr"))
        ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        if ok.sum() < MIN_POINTS or np.ptp(t[ok]) <= 0:
            warnings.warn(f"automatic frequency grid: light curve {name!r} has fewer than {MIN_POINTS} usable "
                          "points and is left out of the signal-to-noise estimate.")
            continue
        try:
            p = fit_light_curve_power(t[ok], y[ok], yerr[ok], smoothing=d["smoothing"])
        except Exception as exc:  # noqa: BLE001 (scipy/linalg failures on odd data)
            warnings.warn(f"automatic frequency grid: the power-spectrum fit of {name!r} failed ({exc}); "
                          "it is left out of the signal-to-noise estimate.")
            continue
        if not all(np.isfinite([p["A"], p["b"], p["c"]])):
            continue
        nz = float(np.sum(0.5 / (yerr[ok] ** 2 + p["c"])))
        fits[name] = p
        terms.append((p["A"], p["b"], p["tau_long"], nz))
    if not terms:
        return None, fits

    def rho_fn(w):
        w = np.asarray(w, dtype=float)
        return sum(A / ((w ** 2 + 1 / a ** 2) * (1.0 + (w * b) ** 2)) * (2 * np.pi / T) * nz
                   for A, b, a, nz in terms)
    return rho_fn, fits


def model_rho(blocks, tau_grid, sigma_drw: float, T: float, tau_drw: Optional[float] = None) -> Callable:
    """rho from a fitted model: ``blocks`` holds each light curve's ``gain``,
    effective noise ``sigma`` and response ``psi`` on ``tau_grid`` (None for
    a driver light curve)."""
    from .forward_model import _filon_weights

    tau = np.asarray(tau_grid, dtype=np.float64)

    def rho_fn(w):
        w = np.asarray(w, dtype=np.float64)
        if tau_drw is None:
            P = sigma_drw ** 2 / w ** 2
        else:
            P = sigma_drw ** 2 * tau_drw / (1.0 + (w * tau_drw) ** 2)
        total = np.zeros_like(w)
        Wc = Ws = None
        for b in blocks:
            if b["psi"] is None:
                H2 = np.ones_like(w)
            else:
                if Wc is None:
                    Wc, Ws = _filon_weights(tau, w, np)
                psi = np.asarray(b["psi"], dtype=np.float64)
                H2 = (Wc @ psi) ** 2 + (Ws @ psi) ** 2
            total += P * (2 * np.pi / T) * b["gain"] ** 2 * H2 * np.sum(0.5 / np.asarray(b["sigma"]) ** 2)
        return total
    return rho_fn


def tabulated(rho_fn: Callable, w_lo: float, w_hi: float, n: int = 2000) -> Callable:
    """``rho_fn`` evaluated once on a logarithmic mesh and interpolated in
    log-log (constant below and above the mesh's ends)."""
    lw = np.linspace(np.log(w_lo), np.log(w_hi), n)
    lr = np.log(np.maximum(rho_fn(np.exp(lw)), 1e-300))
    return lambda w: np.exp(np.interp(np.log(np.asarray(w, dtype=float)), lw, lr))


def capture_blocks(ef, params: dict) -> dict:
    """Each light curve's gain, effective noise and response at ``params``
    (nonlinear sites, constrained), traced from ``ef``'s marginalised model,
    with ``sigma_drw`` and ``tau_drw``: what :func:`model_rho` needs."""
    import jax.numpy as jnp
    from numpyro import handlers

    from . import model

    psis, grabbed = [], {}
    orig_tc, orig_lm = model.transfer_coeffs, model._linear_marginal

    def tc(tau_grid, psi, freqs, matrices=None):
        psis.append(np.asarray(psi, dtype=np.float64))
        return orig_tc(tau_grid, psi, freqs, matrices=matrices)

    def lm(blocks, prior_scale, draw_linear, *rest):
        grabbed["blocks"] = blocks
        return orig_lm(blocks, prior_scale, draw_linear, *rest)

    params = dict(params)
    if "inclination" in params and "cos_inclination" not in params:
        params["cos_inclination"] = float(np.cos(np.deg2rad(params["inclination"])))
    model.transfer_coeffs, model._linear_marginal = tc, lm
    try:
        kwargs = dict(ef._model_kwargs(), marginalise_linear=True)
        sub = {k: jnp.asarray(v) for k, v in params.items()}
        with handlers.seed(rng_seed=0), handlers.substitute(data=sub):
            trace = handlers.trace(model.reverberation_model).get_trace(**kwargs)
    finally:
        model.transfer_coeffs, model._linear_marginal = orig_tc, orig_lm
    blocks, k = [], 0
    for b in grabbed["blocks"]:
        if b["name"] == "driver":
            psi = None
        else:
            psi = psis[k]
            k += 1
        sigma = np.broadcast_to(np.asarray(b["sigma"], dtype=np.float64), np.shape(b["y"]))
        blocks.append(dict(name=b["name"], gain=float(b["gain"]), sigma=sigma, psi=psi))
    return dict(blocks=blocks, sigma_drw=float(trace["sigma_drw"]["value"]),
                tau_drw=float(trace["tau_drw"]["value"]) if "tau_drw" in trace else None)
