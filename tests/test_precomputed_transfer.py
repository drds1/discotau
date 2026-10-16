"""The precomputed trig matrices (``forward_model.transfer_matrices``/
``fourier_basis``) are a pure speed optimisation: they must reproduce the
on-the-fly ``transfer_coeffs``/``compute_echo``/``driver_at`` results, and
the full model's potential energy, to float32 precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import initialize_model

from discotau.echofit import EchoFit
from discotau.forward_model import (
    compute_echo, driver_at, fourier_basis, response_function, transfer_coeffs, transfer_matrices,
)
from discotau.grid_utils import graded_tau_grid
from discotau.model import reverberation_model


def test_transfer_matrices_match_on_the_fly_transfer_coeffs():
    tau_grid = jnp.asarray(graded_tau_grid(60.0, 300, power=3.0), dtype=jnp.float32)
    freqs = jnp.asarray(np.geomspace(0.01, 5.0, 40), dtype=jnp.float32)
    psi = response_function(tau_grid, 0.3, 5000.0, 50.0, 1.0e8)

    A0, B0 = transfer_coeffs(tau_grid, psi, freqs)
    A1, B1 = transfer_coeffs(tau_grid, psi, freqs, matrices=transfer_matrices(tau_grid, freqs))
    np.testing.assert_allclose(A1, A0, atol=1e-5)
    np.testing.assert_allclose(B1, B0, atol=1e-5)


def test_fourier_basis_matches_on_the_fly_echo_and_driver():
    rng = np.random.default_rng(0)
    freqs = jnp.asarray(np.geomspace(0.01, 5.0, 40), dtype=jnp.float32)
    t = jnp.asarray(np.sort(rng.uniform(0.0, 100.0, 50)), dtype=jnp.float32)
    S, C, A, B = (jnp.asarray(rng.normal(size=40), dtype=jnp.float32) for _ in range(4))
    basis = fourier_basis(freqs, t)

    np.testing.assert_allclose(
        compute_echo(S, C, freqs, A, B, t, basis=basis), compute_echo(S, C, freqs, A, B, t), atol=1e-3,
    )
    np.testing.assert_allclose(driver_at(S, C, freqs, t, basis=basis), driver_at(S, C, freqs, t), atol=1e-3)


def test_model_potential_energy_unchanged_by_precomputation():
    rng = np.random.default_rng(1)
    ef = EchoFit(M_BH=1.0e8)
    for name, wav in {"g": 4770.0, "i": 7625.0}.items():
        t = np.sort(rng.uniform(0.0, 80.0, 30))
        ef.add_lightcurve(name, wavelength=wav, t=t, y=rng.normal(size=30), yerr=np.full(30, 0.1))
    ef.add_driver_lightcurve(t=np.sort(rng.uniform(0.0, 80.0, 30)), y=rng.normal(size=30), yerr=np.full(30, 0.1))
    ef.build_grid(n_freq=20, n_tau=120)

    fast = ef._model_kwargs()
    slow = dict(fast, transfer_mats=None, driver={k: v for k, v in fast["driver"].items() if k != "basis"})
    slow["bands"] = {n: {k: v for k, v in d.items() if k != "basis"} for n, d in fast["bands"].items()}

    key = jax.random.PRNGKey(0)
    info_fast = initialize_model(key, reverberation_model, model_kwargs=fast)
    info_slow = initialize_model(key, reverberation_model, model_kwargs=slow)
    z = info_fast.param_info.z
    pe_fast, g_fast = jax.value_and_grad(info_fast.potential_fn)(z)
    pe_slow, g_slow = jax.value_and_grad(info_slow.potential_fn)(z)
    assert float(pe_fast) == pytest.approx(float(pe_slow), rel=1e-4)
    for k in g_fast:
        np.testing.assert_allclose(g_fast[k], g_slow[k], rtol=1e-3, atol=1e-3)
