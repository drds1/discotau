"""
Tests for the driver-amplitude/per-band-gain degeneracy documented in
model.py's module docstring (found via scripts/make_fit_animation.py's GIF,
which made the driver visibly growing/shrinking with little constraint
early in a chain) and its fix, EchoFit._sigma_drw_prior_scale.

test_global_rescale_of_driver_and_gain_is_an_exact_degeneracy mirrors
test_shift_degeneracy.py's style: a direct, no-MCMC proof that rescaling
the driver and compensating every band's gain leaves the predicted light
curve exactly unchanged, independent of lag_mode -- unlike the shift
degeneracy, so no forward-model helper needs a specific response family to
demonstrate it.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from discotau.forward_model import transfer_coeffs, compute_echo, response_function, typical_log_mdot
from discotau.echofit import EchoFit


def test_global_rescale_of_driver_and_gain_is_an_exact_degeneracy():
    rng = np.random.default_rng(0)
    freqs = jnp.asarray(np.geomspace(0.05, 2.0, 20))
    S = jnp.asarray(rng.normal(size=20))
    C = jnp.asarray(rng.normal(size=20))
    tau_grid = jnp.linspace(0.0, 60.0, 400)
    t_obs = jnp.asarray(np.sort(rng.uniform(0.0, 150.0, size=25)))

    psi = response_function(tau_grid, log_mdot_msun_yr=typical_log_mdot(1e8) + 0.1, wavelength=5000.0, inclination=30.0, M_BH=1e8)
    A, B = transfer_coeffs(tau_grid, psi, freqs)

    S_band, C_band = 1.3, 0.4
    y_pred = S_band * compute_echo(S, C, freqs, A, B, t_obs) + C_band

    lam = 7.0  # an arbitrary, non-trivial rescale factor
    y_pred_rescaled = (S_band / lam) * compute_echo(lam * S, lam * C, freqs, A, B, t_obs) + C_band

    assert np.allclose(np.asarray(y_pred), np.asarray(y_pred_rescaled), atol=1e-4), (
        "Rescaling the driver by lambda and every band's gain by 1/lambda changed the "
        "predicted light curve -- either the degeneracy claim is wrong, or compute_echo "
        "changed in a way that broke it."
    )

    # Sanity check the test itself isn't vacuous: rescaling the driver
    # *without* compensating the gain should visibly change the prediction.
    y_pred_uncompensated = S_band * compute_echo(lam * S, lam * C, freqs, A, B, t_obs) + C_band
    assert not np.allclose(np.asarray(y_pred), np.asarray(y_pred_uncompensated), atol=1e-4)


def _echofit_with_bands(band_ys):
    ef = EchoFit(M_BH=1e8)
    t = np.linspace(0.0, 100.0, 20)
    for name, y in band_ys.items():
        yerr = np.full_like(t, 0.1)
        ef.add_lightcurve(name, wavelength=5000.0, t=t, y=np.asarray(y), yerr=yerr)
    return ef


def test_sigma_drw_prior_scale_uses_largest_band_std_without_a_driver():
    rng = np.random.default_rng(0)
    ef = _echofit_with_bands({
        "quiet": rng.normal(0.0, 0.5, size=20),
        "loud": rng.normal(0.0, 5.0, size=20),
    })
    expected = max(np.std(ef.bands["quiet"]["y"]), np.std(ef.bands["loud"]["y"]))
    assert ef._sigma_drw_prior_scale() == pytest.approx(expected)


def test_sigma_drw_prior_scale_prefers_the_driver_light_curve_when_present():
    rng = np.random.default_rng(0)
    ef = _echofit_with_bands({"g": rng.normal(0.0, 0.5, size=20)})
    driver_t = np.linspace(0.0, 100.0, 30)
    driver_y = rng.normal(0.0, 9.0, size=30)
    ef.add_driver_lightcurve(t=driver_t, y=driver_y, yerr=np.full_like(driver_t, 0.1))
    assert ef._sigma_drw_prior_scale() == pytest.approx(np.std(driver_y))


def test_sigma_drw_prior_scale_scales_with_the_data_not_a_fixed_constant():
    y = np.random.default_rng(0).normal(0.0, 0.5, size=20)
    small = _echofit_with_bands({"g": y})
    large = _echofit_with_bands({"g": y * 100.0})
    assert large._sigma_drw_prior_scale() == pytest.approx(100.0 * small._sigma_drw_prior_scale(), rel=1e-6)


def test_model_kwargs_passes_the_data_anchored_scale_through():
    rng = np.random.default_rng(0)
    ef = _echofit_with_bands({"g": rng.normal(0.0, 3.0, size=20)})
    ef.build_grid(n_freq=8, n_tau=40)
    kwargs = ef._model_kwargs()
    assert kwargs["sigma_drw_prior_scale"] == pytest.approx(ef._sigma_drw_prior_scale())
