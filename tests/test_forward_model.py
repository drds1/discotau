import numpy as np
import jax.numpy as jnp

from discotau.forward_model import (
    lag_scaling,
    response_function,
    transfer_coeffs,
    compute_echo,
    driver_at,
)

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid.
_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_lag_scaling_increases_with_wavelength():
    M_BH = 1e8
    log_mdot = 0.0
    tau_short = lag_scaling(log_mdot, 3000.0, M_BH)
    tau_long = lag_scaling(log_mdot, 9000.0, M_BH)
    assert tau_long > tau_short


def test_lag_scaling_increases_with_mdot():
    M_BH = 1e8
    tau_low = lag_scaling(-1.0, 5000.0, M_BH)
    tau_high = lag_scaling(1.0, 5000.0, M_BH)
    assert tau_high > tau_low


def test_response_function_is_causal_and_normalised():
    tau_grid = jnp.linspace(-5.0, 60.0, 500)
    psi = response_function(
        tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=30.0, M_BH=1e8
    )
    psi_np = np.asarray(psi)
    tau_np = np.asarray(tau_grid)
    assert np.all(psi_np[tau_np < 0] == 0.0)
    area = _np_trapz(psi_np, tau_np)
    assert np.isclose(area, 1.0, atol=1e-2)


def test_compute_echo_matches_direct_convolution():
    freqs = jnp.linspace(0.05, 1.0, 8)
    S = jnp.array(np.random.default_rng(0).normal(size=8))
    C = jnp.array(np.random.default_rng(1).normal(size=8))

    tau_grid = jnp.linspace(0.0, 30.0, 2000)
    psi = response_function(
        tau_grid, log_mdot=0.0, wavelength=5000.0, inclination=10.0, M_BH=1e8
    )
    A, B = transfer_coeffs(tau_grid, psi, freqs)

    t_obs = jnp.array([5.0, 12.0, 20.0])
    echo_closed_form = compute_echo(S, C, freqs, A, B, t_obs)

    # brute-force numerical convolution for comparison
    echo_direct = []
    for t in np.asarray(t_obs):
        x_shifted = driver_at(S, C, freqs, jnp.asarray(t - np.asarray(tau_grid)))
        echo_direct.append(_np_trapz(np.asarray(psi) * np.asarray(x_shifted), np.asarray(tau_grid)))
    echo_direct = np.array(echo_direct)

    assert np.allclose(np.asarray(echo_closed_form), echo_direct, atol=1e-2)
