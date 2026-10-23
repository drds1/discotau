"""
The potential's Hessian with the scalar parameters' likelihood block exact.

Most of a fit's nonlinear parameters only rescale things: each light curve's
gain and the driver amplitude sigma_drw (together, each light curve's log scale
``lambda_b = log(gain_b sigma_drw)``) and, with the error model, its error scale
and jitter. For those, :meth:`discotau.gram.GramCache.scalar_hessian` gives the
marginal likelihood's Hessian exactly in about one ``p^3`` product per parameter,
against 2-4 full gradients per parameter for a finite-difference Hessian (83 per
cent of a 3.5 h NGC 5548 single-Gaussian solve). Here it is mapped to NumPyro's
unconstrained coordinates (all of these are log-transformed there, and
``lambda_b = z_S_b + z_sigma_drw`` is linear, so the map adds no second-order
terms), the prior and Jacobian part comes from finite differences of the cheap
prior-only gradient, and every other parameter (accretion rate, inclination,
temperature slope, diffuse continuum, free lags, tau_drw) keeps
finite-difference columns of the full gradient.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import model as _model


class ExactHessian:
    """``hessian(z, sd)`` of the potential ``model(**kwargs)`` at flat
    unconstrained ``z``. Build with :meth:`create`, which returns None when
    the Gram path is not in use (then fall back to finite differences)."""

    @classmethod
    def create(cls, model, kwargs, unravel, z0, full_grad):
        if not _model.gram_in_use(kwargs):
            return None
        return cls(model, kwargs, unravel, z0, full_grad)

    def __init__(self, model, kwargs, unravel, z0, full_grad):
        from numpyro.infer.util import potential_energy

        self.kwargs, self.unravel, self.full_grad = kwargs, unravel, full_grad
        self.cache = kwargs["gram_cache"]
        self.dtype = np.asarray(z0).dtype
        index = unravel(jnp.arange(np.size(z0), dtype=jnp.float32))
        self.index = {k: np.atleast_1d(np.asarray(v)).astype(int) for k, v in index.items()}
        d = int(np.size(z0))
        names = self.cache.names
        # Likelihood coordinates (lambda_b, log s_b, log j_b) -> unconstrained z (linear).
        self.lik_names = None
        sigma = self.index.get("sigma_drw")
        rows = []
        for k, name in enumerate(names):
            r = np.zeros(d)
            if f"S_{name}" in self.index:
                r[self.index[f"S_{name}"][0]] = 1.0
            if sigma is not None:
                r[sigma[0]] = 1.0
            rows.append(("lambda", k, r))
        self._lambda_rows = rows
        scalar = set()
        for name in names:
            for site in (f"S_{name}", f"sigma_scale_{name}", f"sigma_jitter_{name}"):
                if site in self.index:
                    scalar.update(self.index[site].tolist())
        if sigma is not None:
            scalar.update(sigma.tolist())
        self.scalar = np.array(sorted(scalar), dtype=int)
        self.other = np.array([i for i in range(d) if i not in scalar], dtype=int)
        kw0 = dict(kwargs, gram_cache=_model.NULL_LIKELIHOOD)
        self.prior_grad = jax.jit(jax.grad(lambda z: potential_energy(model, (), kw0, unravel(z))))
        self._inputs = jax.jit(self._make_inputs(model, kwargs, unravel))

    @staticmethod
    def _make_inputs(model, kwargs, unravel):
        from numpyro import handlers
        from numpyro.infer.util import constrain_fn

        def inputs(z):
            values = constrain_fn(model, (), kwargs, unravel(z), return_deterministic=False)
            grabbed = {}
            original = _model._linear_marginal

            def grab(blocks, prior_scale, draw_linear, gram_cache=None):
                grabbed["x"] = _model._gram_inputs(blocks, prior_scale)
                return original(blocks, prior_scale, draw_linear, gram_cache)

            _model._linear_marginal = grab
            try:
                handlers.trace(handlers.substitute(handlers.seed(model, 0), values)).get_trace(**kwargs)
            finally:
                _model._linear_marginal = original
            return grabbed["x"]
        return inputs

    def _lik_hessian_z(self, z):
        """The marginal likelihood's Hessian over the scalar parameters, in z."""
        a, b, s, j = (np.asarray(x, dtype=np.float64) for x in self._inputs(jnp.asarray(z, dtype=self.dtype)))
        H, names = self.cache.scalar_hessian(a, b, s, j)
        d = np.size(z)
        J = np.zeros((len(names), d))
        for i, (kind, k) in enumerate(names):
            name = self.cache.names[k]
            if kind == "lambda":
                J[i] = self._lambda_rows[k][2]
            else:
                site = f"sigma_{'scale' if kind == 'log_s' else 'jitter'}_{name}"
                if site in self.index:
                    J[i, self.index[site][0]] = 1.0
        return J.T @ H @ J

    def hessian(self, z, sd, frac: float = 0.1):
        """The potential's Hessian at ``z``; ``sd`` sets the finite-difference
        steps (``frac`` of each coordinate's posterior sd)."""
        z = np.asarray(z, dtype=np.float64)
        d = z.size
        sd = np.asarray(sd if sd is not None else np.ones(d), dtype=np.float64)
        H = np.zeros((d, d))
        Hlik = self._lik_hessian_z(z)
        # Prior and Jacobian part of the scalar block: differences of the prior-only gradient.
        S = self.scalar
        for i in S:
            h = float(np.clip(frac * sd[i], 1e-5, 1.0))
            e = np.zeros(d)
            e[i] = h
            gp = np.asarray(self.prior_grad(jnp.asarray(z + e, dtype=self.dtype)), dtype=np.float64)
            gm = np.asarray(self.prior_grad(jnp.asarray(z - e, dtype=self.dtype)), dtype=np.float64)
            H[S, i] = ((gp - gm) / (2 * h))[S]
        H[np.ix_(S, S)] = 0.5 * (H[np.ix_(S, S)] + H[np.ix_(S, S)].T) - Hlik[np.ix_(S, S)]
        # Every other parameter: finite-difference columns of the full gradient.
        for i in self.other:
            h = float(np.clip(frac * sd[i], 1e-5, 1.0))
            e = np.zeros(d)
            e[i] = h
            col = (np.asarray(self.full_grad(z + e), dtype=np.float64)
                   - np.asarray(self.full_grad(z - e), dtype=np.float64)) / (2 * h)
            H[:, i] = col
            H[i, :] = col
        O = self.other
        if O.size:
            H[np.ix_(O, O)] = 0.5 * (H[np.ix_(O, O)] + H[np.ix_(O, O)].T)
        if not np.all(np.isfinite(H)):
            from .nested_laplace import NonFiniteHessianError

            raise NonFiniteHessianError("exact_hessian: non-finite Hessian")
        return H
