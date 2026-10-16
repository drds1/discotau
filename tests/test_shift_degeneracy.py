"""
Deterministic (no MCMC) test of the identifiability claim documented in
model.py's module docstring: shifting the driver by any Delta, compensated
by shifting every band's response lag by -Delta, leaves the predicted light
curve exactly unchanged for a free-lag (independent, not physically-tied)
response -- which is exactly why such a fit needs a driver light curve to
anchor the absolute lag scale. This checks the forward model directly
(fast, exact, no sampling noise) rather than inferring it indirectly from
an MCMC fit's behaviour.
"""

import numpy as np
import jax.numpy as jnp

from discotau.forward_model import transfer_coeffs, compute_echo


def _fixed_width_tophat(tau_grid, tau_mean, half_width):
    """A top-hat with an *absolute* half-width, independent of tau_mean --
    unlike forward_model.tophat_response_free, whose width scales with
    tau_mean by design (a deliberate astrophysical modelling choice: bigger
    lag, typically broader response). That's an extra effect on top of the
    pure shift degeneracy this test isolates: an exact ``psi'(tau) =
    psi(tau + Delta)`` shift preserves shape *and* width, so this test uses
    a width that doesn't itself change with the shift.
    """
    raw = jnp.where(jnp.abs(tau_grid - tau_mean) <= half_width, 1.0, 0.0)
    raw = jnp.where(tau_grid >= 0.0, raw, 0.0)
    trapz = jnp.trapezoid if hasattr(jnp, "trapezoid") else jnp.trapz
    area = trapz(raw, tau_grid)
    return raw / jnp.clip(area, 1e-12, None)


def _rotate_driver(S, C, freqs, delta):
    """S, C -> the Fourier coefficients of X(t - delta) instead of X(t).

    X(t) = sum_k [S_k sin(w_k t) + C_k cos(w_k t)], so with theta = w_k*delta:
        sin(w_k t - theta) = sin(w_k t) cos(theta) - cos(w_k t) sin(theta)
        cos(w_k t - theta) = cos(w_k t) cos(theta) + sin(w_k t) sin(theta)
    Collecting sin(w_k t)/cos(w_k t) terms gives the rotation below.
    """
    theta = freqs * delta
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    S_shifted = S * cos_t + C * sin_t
    C_shifted = C * cos_t - S * sin_t
    return S_shifted, C_shifted


def test_global_shift_of_driver_and_free_lag_is_an_exact_degeneracy():
    rng = np.random.default_rng(0)
    freqs = jnp.asarray(np.geomspace(0.05, 2.0, 20))
    S = jnp.asarray(rng.normal(size=20))
    C = jnp.asarray(rng.normal(size=20))
    tau_grid = jnp.linspace(0.0, 60.0, 2000)
    t_obs = jnp.asarray(np.sort(rng.uniform(0.0, 150.0, size=25)))

    tau_true = 20.0
    half_width = 4.0
    delta = 6.0  # shift the driver later by 6 days, and the lag earlier by 6 days

    def echo_for(driver_S, driver_C, tau_mean):
        psi = _fixed_width_tophat(tau_grid, tau_mean, half_width)
        A, B = transfer_coeffs(tau_grid, psi, freqs)
        return compute_echo(driver_S, driver_C, freqs, A, B, t_obs)

    echo_unshifted = echo_for(S, C, tau_true)

    S_shifted, C_shifted = _rotate_driver(S, C, freqs, delta)
    echo_shifted = echo_for(S_shifted, C_shifted, tau_true - delta)

    assert np.allclose(np.asarray(echo_unshifted), np.asarray(echo_shifted), atol=1e-2), (
        "A global driver shift + compensating lag shift changed the predicted light curve -- "
        "either the degeneracy claim is wrong, or compute_echo/transfer_coeffs changed in a "
        "way that broke it."
    )

    # Sanity check the test itself isn't vacuous: an *uncompensated* shift
    # (same driver shift, lag left alone) should visibly change the echo.
    echo_uncompensated = echo_for(S_shifted, C_shifted, tau_true)
    assert not np.allclose(
        np.asarray(echo_unshifted), np.asarray(echo_uncompensated), atol=1e-2
    )
