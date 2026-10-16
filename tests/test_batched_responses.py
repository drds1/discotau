"""
model._call_responses: every physical band's response in one jax.vmap'd call.
It must give each band exactly what a per-band call gives, and fall back to
per-band calls for a response that can't be mapped over a traced wavelength.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from discotau import model
from discotau.forward_model import response_function, thin_disk_response

TAU = jnp.concatenate([jnp.zeros(1), jnp.geomspace(0.01, 30.0, 199)])
WAVELENGTHS = [1928.0, 3600.0, 4770.0, 7625.0]


def _per_band(fn, **kw):
    return [np.asarray(model._call_response(fn, TAU, 0.3, wl, 40.0, 1e8, **kw)) for wl in WAVELENGTHS]


def test_batched_thin_disc_responses_match_per_band_calls():
    for fn, kw in [(thin_disk_response, {}), (thin_disk_response, dict(viscous_slope=jnp.float32(0.9))),
                   (functools.partial(thin_disk_response, include_irradiation=False), {})]:
        batched = model._call_responses(fn, TAU, 0.3, WAVELENGTHS, 40.0, 1e8, **kw)
        for b, s in zip(batched, _per_band(fn, **kw)):
            np.testing.assert_allclose(np.asarray(b), s, rtol=1e-4, atol=1e-6 * s.max())


def test_batched_gradient_matches_per_band_gradient():
    def loss(p, batched):
        kw = dict(viscous_slope=p[2])
        if batched:
            psis = model._call_responses(thin_disk_response, TAU, p[0], WAVELENGTHS, p[1], 1e8, **kw)
        else:
            psis = [model._call_response(thin_disk_response, TAU, p[0], wl, p[1], 1e8, **kw) for wl in WAVELENGTHS]
        return sum(jnp.sum(psi ** 2) for psi in psis)

    p = jnp.array([0.3, 40.0, 0.8])
    np.testing.assert_allclose(np.asarray(jax.grad(loss)(p, True)), np.asarray(jax.grad(loss)(p, False)), rtol=1e-3)


def test_skew_normal_response_is_batched_too():
    batched = model._call_responses(response_function, TAU, 0.3, WAVELENGTHS, 40.0, 1e8)
    for b, s in zip(batched, _per_band(response_function)):
        np.testing.assert_allclose(np.asarray(b), s, rtol=1e-5, atol=1e-7)


def test_a_response_that_cannot_be_mapped_falls_back_to_per_band_calls():
    def numpy_response(tau_grid, log_mdot, wavelength, inclination, M_BH):
        # float() of a traced wavelength fails under jax.vmap / jax.jit.
        centre = 0.001 * float(wavelength)
        psi = np.exp(-0.5 * ((np.asarray(tau_grid) - centre) / 0.5) ** 2)
        return jnp.asarray(psi / np.trapezoid(psi, np.asarray(tau_grid)) if hasattr(np, "trapezoid")
                           else psi / np.trapz(psi, np.asarray(tau_grid)))

    batched = model._call_responses(numpy_response, TAU, 0.3, WAVELENGTHS, 40.0, 1e8)
    for psi, wl in zip(batched, WAVELENGTHS):
        assert abs(float(TAU[int(np.argmax(np.asarray(psi)))]) - 0.001 * wl) < 0.2
