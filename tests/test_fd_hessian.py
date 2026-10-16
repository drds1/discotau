"""The finite-difference Hessian behind both direct solves' posterior widths
(nested_laplace.fd_hessian, CLAUDE.md decision #30): on a potential whose
curvatures differ by 3e4, like the driver-amplitude/band-gain ridge, its
float32 gradients must still give every width to within a couple of per cent,
measured along the eigenvectors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from discotau.nested_laplace import fd_hessian

EIGENVALUES = np.array([3.0e4, 50.0, 1.0, 1.0])


def _quadratic(seed=0):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a = rotation @ np.diag(EIGENVALUES) @ rotation.T
    a32 = jnp.asarray(a, dtype=jnp.float32)
    grad = jax.jit(jax.grad(lambda z: 0.5 * z @ a32 @ z))
    return a, (lambda z: grad(jnp.asarray(z, dtype=jnp.float32))), rng.normal(size=4)


def test_widths_match_the_exact_hessian_on_an_ill_conditioned_quadratic():
    a, grad, z = _quadratic()
    h, vals, vecs = fd_hessian(grad, z, sd=np.ones(4))
    exact_widths = np.sqrt(np.diag(np.linalg.inv(a)))
    widths = np.sqrt(np.diag(np.linalg.inv(h)))
    np.testing.assert_allclose(widths, exact_widths, rtol=0.02)
    np.testing.assert_allclose(np.sort(vals), np.sort(EIGENVALUES), rtol=0.02)
    np.testing.assert_allclose(h, h.T)


def test_reusing_an_eigenbasis_gives_the_same_hessian():
    a, grad, z = _quadratic(seed=1)
    _, vals, vecs = fd_hessian(grad, z, sd=np.ones(4))
    h2, vals2, _ = fd_hessian(grad, z, basis=(vals, vecs))
    np.testing.assert_allclose(np.sort(vals2), np.sort(EIGENVALUES), rtol=0.02)
    np.testing.assert_allclose(np.sqrt(np.diag(np.linalg.inv(h2))), np.sqrt(np.diag(np.linalg.inv(a))), rtol=0.02)
