"""
Regression test for CLAUDE.md design decision #5 ("response function is
swappable by contract"): swapping ``discotau.model.response_function`` must
be honoured by *both* fitting and ``EchoFit.plot_lightcurve_fits()``.

It previously wasn't -- ``echofit.py`` had its own independent
``from .forward_model import response_function`` import, so a swap on
``model.py`` changed what NUTS fit but not what the plot re-derived psi
from, silently producing a plot inconsistent with the actual fit.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import discotau.model as model_mod
from discotau.forward_model import lag_scaling
from discotau.synthetic import generate_synthetic_dataset
from discotau.echofit import EchoFit


def _tophat_response(tau_grid, log_mdot_msun_yr, wavelength, inclination, M_BH, width_frac=0.3):
    """A flat-topped boxcar response -- unlike the default skew-normal,
    its values near the peak are (near-)identical, which is what the test
    below checks for."""
    tau_mean = lag_scaling(log_mdot_msun_yr, wavelength, M_BH)
    half_width = jnp.clip(width_frac * tau_mean, 1e-3, None)
    raw = jnp.where(jnp.abs(tau_grid - tau_mean) <= half_width, 1.0, 0.0)
    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    return raw / jnp.clip(area, 1e-12, None)


def _fit_and_get_psi_median(seed=0):
    data = generate_synthetic_dataset(
        M_BH=1.0e9, bands={"g": 4770.0}, n_obs_per_band=20,
        n_freq=8, n_tau=300, tau_max=50.0, noise_level=0.08, seed=seed,
    )
    ef = EchoFit(M_BH=1.0e9)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=300)
    ef.fit(rng_seed=0, num_warmup=30, num_samples=30, progress_bar=False)

    fig, axes = ef.plot_lightcurve_fits(n_fine=30, n_pred_samples=10)
    ax_psi = axes[1, 1]  # row 1 = the single band's row; col 1 = psi panel
    (line,) = ax_psi.get_lines()
    return line.get_ydata(), np.asarray(ef.tau_grid)


def _kurtosis(psi, tau_grid):
    """Excess-kurtosis-free (raw) kurtosis of psi as a probability density
    on tau_grid -- a global shape descriptor (integrates over the whole
    curve) rather than a handful of near-peak grid points, so it isn't
    sensitive to this test's short 30+30-sample chain not converging
    tightly (the plotted psi is a *median across posterior draws*, so a
    loosely-converged log_mdot_msun_yr/inclination posterior adds a little
    point-to-point noise a purely local check would be sensitive to). A
    uniform/boxcar distribution's kurtosis is 1.8; a skew-normal's is much
    closer to a Gaussian's 3 -- a large, robust separation.
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    p = psi / trapz(psi, tau_grid)
    mean = trapz(p * tau_grid, tau_grid)
    var = trapz(p * (tau_grid - mean) ** 2, tau_grid)
    fourth = trapz(p * (tau_grid - mean) ** 4, tau_grid)
    return fourth / var ** 2


@pytest.mark.fit
@pytest.mark.slow
def test_swapping_model_response_function_is_reflected_in_the_plot():
    original = model_mod.response_function
    try:
        default_psi, tau_grid = _fit_and_get_psi_median()

        model_mod.response_function = _tophat_response
        tophat_psi, _ = _fit_and_get_psi_median()
    finally:
        model_mod.response_function = original

    # The direct, robust check of the actual regression (CLAUDE.md decision
    # #5): if plot_lightcurve_fits() silently re-derived psi from the
    # *original* response_function instead of the swapped one, tophat_psi
    # would look like default_psi (both the skew-normal), not the tophat.
    assert not np.allclose(default_psi, tophat_psi, atol=1e-3), (
        "psi looks the same before and after swapping model.response_function -- "
        "plot_lightcurve_fits() may be re-deriving psi from the original "
        "response_function again instead of the swapped one."
    )

    # And it should actually look tophat-like (flat, kurtosis ~1.8), not
    # skew-normal-like (peaked, kurtosis closer to a Gaussian's 3) -- a
    # global-shape check, not a local one, so it isn't noise-sensitive the
    # way comparing a handful of near-peak grid points was (see git history
    # / CLAUDE.md decision #13 for that fragility, exposed by anchoring
    # sigma_drw's prior to the data shifting this test's specific fit
    # enough to tip a threshold tuned against the old prior's behaviour).
    assert _kurtosis(tophat_psi, tau_grid) < _kurtosis(default_psi, tau_grid) - 0.3
