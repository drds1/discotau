"""
build_grid()'s driver frequency grids: the fixed one (harmonics of the longest
period, then log spacing, up to f_max cycles per day; CLAUDE.md decision #29),
now build_grid(log_step=...), and the automatic default (decision #38), with
its post-fit check; and the default linear background. No fits.
"""

import numpy as np
import pytest

from discotau import run_manager
from discotau.echofit import DEFAULT_BACKGROUND_ORDER, F_MAX_CYCLES_PER_DAY, FREQ_LOG_STEP, EchoFit
from discotau import frequency_grid
from discotau.echofit import PERIOD_MAX_FACTOR
from discotau.grid_utils import hybrid_frequency_grid


def _echofit(t_end=100.0):
    t = np.arange(0.0, t_end, 1.0)
    ef = EchoFit(M_BH=1e8)
    ef.add_lightcurve("g", wavelength=4770.0, t=t, y=np.sin(t / 10.0), yerr=np.full_like(t, 0.1))
    return ef


def test_hybrid_grid_is_harmonic_then_logarithmic():
    period_max, f_max, log_step = 300.0, 2.0, 0.03
    w = hybrid_frequency_grid(period_max, f_max, log_step)
    dw = 2.0 * np.pi / period_max
    steps = np.diff(w)
    assert np.isclose(w[0], dw)
    assert np.all(steps > 0)
    assert w[-1] <= 2.0 * np.pi * f_max * (1 + 1e-9)
    assert w[-1] * (1 + log_step) > 2.0 * np.pi * f_max  # one more step would overshoot
    linear = w[:-1] * log_step < dw
    assert linear.any() and (~linear).any()
    # Harmonics of the longest period below the crossover, a constant ratio above it.
    assert np.allclose(steps[linear], dw)
    assert np.allclose(steps[~linear] / w[:-1][~linear], log_step)
    assert np.allclose(w[: linear.sum() + 1] / dw, np.arange(1, linear.sum() + 2))


def test_hybrid_grid_needs_far_fewer_modes_than_linear_spacing():
    # An NGC 5548-length campaign: linear spacing to 2 cycles/day needs ~2 period_max modes.
    period_max = 2.0 * (289.0 + 144.5)
    n = len(hybrid_frequency_grid(period_max, 2.0, 0.03))
    assert n < 200 < 2.0 * period_max


def test_hybrid_grid_rejects_non_positive_settings():
    with pytest.raises(ValueError):
        hybrid_frequency_grid(100.0, 0.0, 0.03)


def test_fixed_grid_reaches_f_max_whatever_the_cadence():
    for cadence_end in (100.0, 400.0):
        ef = _echofit(cadence_end)
        ef.build_grid(n_tau=30, log_step=FREQ_LOG_STEP)
        w = np.asarray(ef.freqs)
        f_top = w.max() / (2.0 * np.pi)
        assert F_MAX_CYCLES_PER_DAY / (1 + FREQ_LOG_STEP) < f_top <= F_MAX_CYCLES_PER_DAY
        assert np.isclose(w.min(), 2.0 * np.pi / (2.0 * (cadence_end - 1.0 + 0.5 * (cadence_end - 1.0))))


def test_build_grid_f_max_and_dt_min_set_the_upper_frequency():
    ef = _echofit()
    ef.build_grid(n_tau=30, f_max=0.5, log_step=FREQ_LOG_STEP)
    assert np.isclose(float(ef.freqs.max()) / (2 * np.pi), 0.5, rtol=0.03)
    ef.build_grid(n_tau=30, dt_min=0.25, log_step=FREQ_LOG_STEP)  # w_max = pi / dt_min, i.e. 2 cycles/day
    assert np.isclose(float(ef.freqs.max()) / (2 * np.pi), 2.0, rtol=0.03)


def test_build_grid_n_freq_keeps_the_old_log_grid():
    ef = _echofit()
    ef.build_grid(n_tau=30, n_freq=12, dt_min=1.0)
    w = np.asarray(ef.freqs)
    assert len(w) == 12
    assert np.isclose(w.max(), np.pi / 1.0)
    assert np.allclose(np.diff(np.log(w)), np.log(w[1] / w[0]))


def test_light_curves_get_a_linear_background_by_default():
    ef = _echofit()
    ef.add_driver_lightcurve(t=[0.0, 1.0, 2.0], y=[1.0, 2.0, 1.0], yerr=[0.1, 0.1, 0.1])
    assert DEFAULT_BACKGROUND_ORDER == 1
    assert ef.bands["g"]["background_order"] == 1
    assert ef.driver_data["background_order"] == 1


def test_checkpoints_saved_before_the_background_default_resume_without_one(tmp_path):
    # A checkpoint written before background_order existed has no such key; it
    # must reload as 0 (its model), not as today's default.
    t = np.arange(5.0)
    np.savez(tmp_path / "bands.npz", names=np.array(["g"]), g__t=t, g__y=t, g__yerr=t + 1, g__wavelength=np.asarray(4770.0))
    np.savez(tmp_path / "driver.npz", t=t, y=t, yerr=t + 1)
    assert run_manager.load_bands_npz(tmp_path / "bands.npz")["g"]["background_order"] == 0
    assert run_manager.load_driver_npz(tmp_path / "driver.npz")["background_order"] == 0


def test_fixed_grid_log_step_is_one_per_cent():
    """The fixed grid's step was 0.03 until October 2026; 0.01 is converged on a
    649-day campaign (CLAUDE.md decision #35)."""
    assert FREQ_LOG_STEP == 0.01
    ef = _echofit(400.0)
    ef.build_grid(n_tau=30, log_step=FREQ_LOG_STEP)
    w = np.asarray(ef.freqs)
    steps = np.diff(w)
    log_part = steps > 1.0001 * (2.0 * np.pi / (2.0 * (399.0 + 0.5 * 399.0)))
    assert np.allclose(steps[log_part] / w[:-1][log_part], 0.01, rtol=1e-3)  # float32 frequencies


# ---------------------------------------------------------------- automatic grid
def _red_noise_echofit(snr, T=200.0, bands=("g", "i"), seed=0, M_BH=1e8):
    """Light curves of a random walk smoothed over ~2 days, sampled daily, with
    noise 1/snr of the variability's scale."""
    rng = np.random.default_rng(seed)
    dt = 0.05
    t_fine = np.arange(-50.0, T, dt)
    x = np.cumsum(rng.normal(size=len(t_fine))) * np.sqrt(dt)
    kernel = np.exp(-np.arange(0.0, 10.0, dt) / 2.0)
    echo = np.convolve(x, kernel / kernel.sum())[: len(x)]
    ef = EchoFit(M_BH=M_BH)
    for k, (name, lam) in enumerate(zip(bands, (4770.0, 7545.0, 3465.0))):
        t = np.sort(rng.uniform(0.0, T, int(T)))
        f = np.interp(t, t_fine, echo)
        scale = np.std(f)
        yerr = np.full(len(t), scale / snr)
        ef.add_lightcurve(name, wavelength=lam, t=t, y=f + yerr * rng.normal(size=len(t)), yerr=yerr)
    return ef


def test_automatic_grid_is_the_default_and_has_the_designed_structure():
    ef = _red_noise_echofit(snr=100.0)
    ef.build_grid(n_tau=30, tau_max=40.0)
    w = np.asarray(ef.freqs, dtype=np.float64)
    info = ef.grid_info
    assert info["kind"] == "auto" and info["n_modes"] == len(w)
    t = np.concatenate([d["t"] for d in ef.bands.values()])
    span = t.max() - t.min()
    h = 2 * np.pi / (PERIOD_MAX_FACTOR * (span + 40.0))
    assert np.isclose(w[0], h, rtol=1e-5)
    n_trend = frequency_grid.N_TREND_MODES
    assert np.allclose(w[:n_trend] / h, np.arange(1, n_trend + 1), rtol=1e-5)
    steps = np.diff(w)[n_trend - 1:]
    # 1e-4: the frequencies are float32, so a step near 1 cycle/day is rounded by ~2e-5 of itself.
    floor = 2 * np.pi / (span + 40.0)
    assert np.all(steps >= floor * (1 - 1e-4))                            # never denser than the window
    # never sparser than 3 per cent of w, except where the window's floor is wider still
    assert np.all(steps <= np.maximum(floor, frequency_grid.MAX_LOG_STEP * w[n_trend - 1:-1]) * (1 + 1e-4))
    f_top = w[-1] / (2 * np.pi)
    assert F_MAX_CYCLES_PER_DAY * 0.97 <= f_top <= frequency_grid.F_CAP_FACTOR * F_MAX_CYCLES_PER_DAY
    assert set(info["light_curves"]) == {"g", "i"}


def test_automatic_grid_grows_with_signal_to_noise():
    sizes = []
    for snr in (10.0, 100.0, 1000.0):
        ef = _red_noise_echofit(snr=snr)
        ef.build_grid(n_tau=30, tau_max=40.0)
        sizes.append(len(ef.freqs))
    assert sizes[0] < sizes[1] < sizes[2]


def test_automatic_grid_safety_factor_adds_modes():
    ef = _red_noise_echofit(snr=100.0)
    ef.build_grid(n_tau=30, tau_max=40.0, grid_safety=1.0)
    n1 = len(ef.freqs)
    ef.build_grid(n_tau=30, tau_max=40.0, grid_safety=10.0)
    assert len(ef.freqs) > n1


def test_automatic_grid_falls_back_to_the_fixed_grid_without_usable_light_curves():
    ef = EchoFit(M_BH=1e8)
    t = np.arange(5.0)
    ef.add_lightcurve("g", wavelength=4770.0, t=t, y=np.sin(t), yerr=np.full(5, 0.1))
    with pytest.warns(UserWarning, match="fixed grid"):
        ef.build_grid(n_tau=30)
    assert ef.grid_info["kind"] == "fixed" and ef.grid_info["fallback"]


def test_power_fit_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 100, 80))
    y, yerr = np.cumsum(rng.normal(size=80)) * 0.1, np.full(80, 0.05)
    D = t[:, None] - t[None, :]
    for smoothing, th in ((True, np.array([-3.0, 0.5, -7.0])), (False, np.array([-3.0, -7.0]))):
        _, g = frequency_grid._gp_nll_grad(th, D, y - y.mean(), yerr ** 2, 200.0, smoothing)
        fd = [(frequency_grid._gp_nll_grad(th + 1e-5 * e, D, y - y.mean(), yerr ** 2, 200.0, smoothing)[0]
               - frequency_grid._gp_nll_grad(th - 1e-5 * e, D, y - y.mean(), yerr ** 2, 200.0, smoothing)[0]) / 2e-5
              for e in np.eye(len(th))]
        assert np.allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_loss_density_is_zero_at_or_below_the_window_resolution():
    r = np.array([0.0, 0.1, 10.0, 1e3])
    assert np.allclose(frequency_grid.loss_density(r, 1.0), 0.0)
    assert np.allclose(frequency_grid.loss_density(r, 0.3), 0.0)
    assert np.all(np.diff(frequency_grid.loss_density(10.0, np.array([1.0, 2.0, 4.0, 8.0]))) > 0)


def test_check_frequency_grid_flags_a_coarse_grid_and_suggests_a_finer_one():
    ef = _red_noise_echofit(snr=300.0, bands=("g", "i", "u"))
    ef.build_grid(n_tau=60, tau_max=40.0, n_freq=20)
    params = {"log_mdot_msun_yr": -0.65, "inclination": 30.0, "sigma_drw": 1.0}
    params.update({f"S_{n}": float(np.std(d["y"])) for n, d in ef.bands.items()})
    with pytest.warns(UserWarning, match="check_frequency_grid"):
        check = ef.check_frequency_grid(params=params)
    assert check["predicted_loss"] > frequency_grid.CHECK_LOSS
    assert check["suggested_n_modes"] > 20
    ef.freqs = np.asarray(check["suggested_freqs"], dtype=np.float32)
    assert ef.check_frequency_grid(params=params, warn=False)["predicted_loss"] < frequency_grid.CHECK_LOSS
