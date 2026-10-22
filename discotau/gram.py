"""
The marginal likelihood of the linear parameters from precomputed Gram
matrices, in float64 on the host.

With every nonlinear parameter fixed, light curve ``b`` is

    y_b = Phi_b R_b theta + E_b e + noise,   noise ~ N(0, D_b),

with ``Phi_b = [sin(w t), cos(w t)]`` at its times (fixed for the whole fit),
``theta`` the unit-normal driver coefficients (``S_raw``, ``C_raw``), ``e`` the
unit-normal offsets and background coefficients, and ``R_b`` block-diagonal in
the modes: for mode k, ``[[a_k, b_k], [-b_k, a_k]]`` mapping (S_k, C_k) to the
(sin, cos) columns, with ``a + i b = gain_b * prior_sd_k * (A_k + i B_k)``,
the response's transfer coefficients (1 and 0 for a driver light curve). Only
``R_b`` and the noise depend on the nonlinear parameters, so with the Gram
matrix ``G_b = Phi_b^T D_b^-1 Phi_b`` and its offset/background companions
computed once,

    P = I + sum_b R_b^T G_b R_b   (+ the E rows and columns),
    log p(y) = -1/2 [y^T D^-1 y - c^T P^-1 c + log det P + log det D + N log 2 pi],

``c`` the projections of the data. One evaluation costs O(B p^2 + p^3) for p
linear parameters and B light curves, independent of the number of data points,
against O((N + p) p^2) for the QR of ``model._marginal_solve``. Evaluated in
float64 (LAPACK), the textbook quadratic form is safe; it cancels
catastrophically in float32, which is why the QR form was used before
(CLAUDE.md decision #21).

The error model (``sigma_i^2 = s^2 e_i^2 + j^2``) changes ``D_b``. The weights
are ``(1/s^2) / (e_i^2 + u)`` with ``u = j^2 / s^2``, and the family
``u -> 1 / (e_i^2 + u)`` has low numerical rank over one light curve's quoted
errors (10-12 terms to 1e-9 over seven decades of u on Fairall 9's Swift
bands), the data-adaptive relative of exponential-sum approximations of 1/x
(Beylkin & Monzon 2005). With an orthonormal basis ``U`` of the family, the
weights at any ``u`` project exactly onto ``r`` coefficients, and every Gram
quantity is the same combination of ``r`` precomputed ones.

Gradients: with ``u = P^-1 c`` and ``W = P^-1 + u u^T``,
``dL/dR_b = h_b u^T - G_b R_b W - G_E,b W_E``, of which only the 2x2 diagonal
sub-blocks matter, and the error-model parameters through the ``r``
coefficients. :func:`GramCache.loglik_fn` wraps it as a JAX function (a
``pure_callback`` with a ``custom_vjp``) of each light curve's ``(a, b)``,
error scale and jitter, so everything upstream stays in JAX.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

# Weight-family basis: singular values kept down to this fraction of the
# largest, over u from U_SPAN[0] to U_SPAN[1] times the mean squared error
# (plus u = 0).
WEIGHT_RANK_TOL = 1e-10
U_SPAN = (1e-6, 1e4)


class GramCache:
    """Gram matrices of every light curve for one frequency grid, built on
    first use. ``curves``: block name -> dict with ``t``, ``y``, ``yerr``,
    ``error_model`` (bool), ``offset_loc``, ``offset_sd``, ``bg_cols`` (``(n, K)``
    or None) and ``bg_sd``, in the order ``model._linear_marginal`` stacks
    them (the driver light curve first)."""

    def __init__(self, freqs, curves: dict):
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.curves = curves
        self.names = list(curves)
        self.K = len(self.freqs)
        self.n_off = len(curves)
        self.n_bg = sum(0 if c["bg_cols"] is None else c["bg_cols"].shape[1] for c in curves.values())
        self.p = 2 * self.K + self.n_off + self.n_bg
        self._blocks = None
        self._jax_fns = None

    # ------------------------------------------------------------ set-up
    def _weight_basis(self, e2, error_model):
        if not error_model:
            return (1.0 / e2)[:, None], False
        m = float(np.mean(e2))
        u = np.concatenate([[0.0], np.geomspace(U_SPAN[0] * m, U_SPAN[1] * m, 300)])
        F = 1.0 / (e2[:, None] + u[None, :])
        U, S, _ = np.linalg.svd(F / np.linalg.norm(F, axis=0), full_matrices=False)
        r = max(1, int(np.sum(S > WEIGHT_RANK_TOL * S[0])))
        return U[:, :r], True

    def build(self):
        if self._blocks is not None:
            return self
        w = self.freqs
        blocks, bg0 = [], 2 * self.K + self.n_off
        for i, (name, c) in enumerate(self.curves.items()):
            t = np.asarray(c["t"], dtype=np.float64)
            y = np.asarray(c["y"], dtype=np.float64)
            e2 = np.asarray(c["yerr"], dtype=np.float64) ** 2
            yt = y - float(c["offset_loc"])
            cols = [np.full(len(t), float(c["offset_sd"]))]
            idx = [2 * self.K + i]
            if c["bg_cols"] is not None:
                kb = c["bg_cols"].shape[1]
                cols += list((np.asarray(c["bg_cols"], dtype=np.float64) * float(c["bg_sd"])).T)
                idx += list(range(bg0, bg0 + kb))
                bg0 += kb
            E = np.column_stack(cols)
            wt = np.outer(t, w)
            Phi = np.hstack([np.sin(wt), np.cos(wt)])
            U, projected = self._weight_basis(e2, c["error_model"])
            G, GE, EE, h, hE, yy = [], [], [], [], [], []
            for r in range(U.shape[1]):
                d = U[:, r]
                PhiD = Phi * d[:, None]
                G.append(PhiD.T @ Phi)
                GE.append(PhiD.T @ E)
                EE.append((E * d[:, None]).T @ E)
                h.append(PhiD.T @ yt)
                hE.append((E * d[:, None]).T @ yt)
                yy.append(float(yt @ (d * yt)))
            G = np.stack(G)
            K = self.K
            blocks.append(dict(name=name, e2=e2, U=U, projected=projected, e_idx=np.array(idx),
                               G=G, Gss=np.ascontiguousarray(G[:, :K, :K]), Gsc=np.ascontiguousarray(G[:, :K, K:]),
                               Gcc=np.ascontiguousarray(G[:, K:, K:]),
                               GE=np.stack(GE), EE=np.stack(EE), h=np.stack(h), hE=np.stack(hE), yy=np.array(yy)))
        self._blocks = blocks
        self.N = int(sum(len(b["e2"]) for b in blocks))
        return self

    # ------------------------------------------------------------ numerics
    def _coeffs(self, blk, s, j):
        """Weight coefficients q_r (the Gram quantities are sum_r q_r X_r) and
        their derivatives with respect to s and j."""
        if not blk["projected"]:
            return np.ones(1), np.zeros(1), np.zeros(1)
        u = (j / s) ** 2
        wv = 1.0 / (blk["e2"] + u)
        c = blk["U"].T @ wv
        dc_du = blk["U"].T @ (-wv * wv)
        q = c / s ** 2
        dq_ds = -2.0 * c / s ** 3 + dc_du * (-2.0 * j ** 2 / s ** 3) / s ** 2
        dq_dj = dc_du * (2.0 * j / s ** 2) / s ** 2
        return q, dq_ds, dq_dj

    def _assemble(self, a, b, s, j):
        """P (the linear parameters' conditional precision), the data
        projections, and per-block pieces the gradient reuses. R^T G R is
        accumulated as four K x K blocks: with G R's blocks
        GR_sS = G_ss a - G_sc b, GR_sC = G_ss b + G_sc a (and cos rows alike),
        P_SS = a GR_sS - b GR_cS, P_SC = a GR_sC - b GR_cC,
        P_CS = b GR_sS + a GR_cS, P_CC = b GR_sC + a GR_cC
        (a, b scaling columns, then rows), in place."""
        K, p = self.K, self.p
        P = np.zeros((p, p))
        cvec = np.zeros(p)
        yy = logdetD = 0.0
        parts = []
        PSS, PSC, PCS, PCC = P[:K, :K], P[:K, K:2 * K], P[K:2 * K, :K], P[K:2 * K, K:2 * K]
        tmp = np.empty((K, K))
        for k, blk in enumerate(self._blocks):
            q, dq_ds, dq_dj = self._coeffs(blk, s[k], j[k])
            if blk["projected"]:
                Gss, Gsc, Gcc = (np.tensordot(q, blk[n], 1) for n in ("Gss", "Gsc", "Gcc"))
            else:
                Gss, Gsc, Gcc = blk["Gss"][0], blk["Gsc"][0], blk["Gcc"][0]
            GE = np.tensordot(q, blk["GE"], 1)
            ak, bk = a[k], b[k]
            Gcs = Gsc.T
            GR_sS = Gss * ak - Gsc * bk
            GR_sC = Gss * bk + Gsc * ak
            GR_cS = Gcs * ak - Gcc * bk
            GR_cC = Gcs * bk + Gcc * ak
            ac, bc = ak[:, None], bk[:, None]
            np.multiply(GR_sS, ac, out=tmp); PSS += tmp
            np.multiply(GR_cS, bc, out=tmp); PSS -= tmp
            np.multiply(GR_sC, ac, out=tmp); PSC += tmp
            np.multiply(GR_cC, bc, out=tmp); PSC -= tmp
            np.multiply(GR_sS, bc, out=tmp); PCS += tmp
            np.multiply(GR_cS, ac, out=tmp); PCS += tmp
            np.multiply(GR_sC, bc, out=tmp); PCC += tmp
            np.multiply(GR_cC, ac, out=tmp); PCC += tmp
            RtGE = _rt(ak, bk, GE)
            ei = blk["e_idx"]
            P[:2 * K, ei] += RtGE
            P[ei, :2 * K] += RtGE.T
            P[np.ix_(ei, ei)] += np.tensordot(q, blk["EE"], 1)
            h = q @ blk["h"]
            cvec[:2 * K] += _rt(ak, bk, h[:, None])[:, 0]
            cvec[ei] += q @ blk["hE"]
            yy += float(q @ blk["yy"])
            sig2 = s[k] ** 2 * blk["e2"] + j[k] ** 2
            logdetD += float(np.sum(np.log(sig2)))
            parts.append(dict(q=q, dq_ds=dq_ds, dq_dj=dq_dj, GR=(GR_sS, GR_sC, GR_cS, GR_cC), GE=GE, h=h, sig2=sig2))
        P[np.diag_indices(p)] += 1.0
        return P, cvec, yy, logdetD, parts

    def _value(self, a, b, s, j):
        P, cvec, yy, logdetD, _ = self._assemble(a, b, s, j)
        cf = _cholesky(P)
        if cf is None:
            return np.nan
        u = sla.cho_solve(cf, cvec, check_finite=False)
        logdetP = 2.0 * np.sum(np.log(np.diag(cf[0])))
        return -0.5 * (yy - cvec @ u + logdetP + logdetD + self.N * np.log(2.0 * np.pi))

    def value_and_grad(self, a, b, s, j):
        """Log likelihood and its gradient with respect to every block's
        per-mode ``a``, ``b`` (``(B, K)``), error scale ``s`` and jitter ``j``
        (``(B,)``; 1 and 0 where a block has no error model)."""
        self.build()
        a, b, s, j = (np.asarray(x, dtype=np.float64) for x in (a, b, s, j))
        K = self.K
        P, cvec, yy, logdetD, parts = self._assemble(a, b, s, j)
        cf = _cholesky(P)
        if cf is None:
            # Not numerically positive definite (an absurd trial point, e.g. a
            # line search stepping to an enormous driver amplitude): report a
            # non-finite likelihood, which every solver already treats as a
            # point to back away from, instead of raising inside the callback.
            return np.nan, np.zeros_like(a), np.zeros_like(b), np.zeros_like(s), np.zeros_like(j)
        u = sla.cho_solve(cf, cvec, check_finite=False)
        logdetP = 2.0 * np.sum(np.log(np.diag(cf[0])))
        L = -0.5 * (yy - cvec @ u + logdetP + logdetD + self.N * np.log(2.0 * np.pi))
        W = sla.cho_solve(cf, np.eye(self.p), check_finite=False)
        W += np.outer(u, u)
        Wtt, ut = W[:2 * K, :2 * K], u[:2 * K]
        da, db = np.zeros_like(a), np.zeros_like(b)
        ds, dj = np.zeros_like(s), np.zeros_like(j)
        for k, (blk, pt) in enumerate(zip(self._blocks, parts)):
            ei = blk["e_idx"]
            Wet = W[ei, :2 * K]
            (GR_sS, GR_sC, GR_cS, GR_cC), GE, h = pt["GR"], pt["GE"], pt["h"]
            # dL/dR = h u^T - G R W_tt - G_E W_et: diagonals of its four K x K
            # sub-blocks. (G R W)_kk over (row r, col c) = sum over the S and C
            # halves of GR_rS[k] . W_S,c[:, k] + GR_rC[k] . W_C,c[:, k].
            WS, WC = Wtt[:K], Wtt[K:]

            def dblk(GR_rS, GR_rC, rows, cols):
                return (h[rows] * ut[cols]
                        - np.einsum("kj,jk->k", GR_rS, WS[:, cols]) - np.einsum("kj,jk->k", GR_rC, WC[:, cols])
                        - np.einsum("kj,jk->k", GE[rows], Wet[:, cols]))
            sl_s, sl_c = slice(0, K), slice(K, 2 * K)
            sS = dblk(GR_sS, GR_sC, sl_s, sl_s)
            sC = dblk(GR_sS, GR_sC, sl_s, sl_c)
            cS = dblk(GR_cS, GR_cC, sl_c, sl_s)
            cC = dblk(GR_cS, GR_cC, sl_c, sl_c)
            da[k] = sS + cC
            db[k] = sC - cS
            if blk["projected"]:
                ak, bk = a[k], b[k]
                # dL/dq_r: every Gram quantity is linear in q.
                RWRt = _r(ak, bk, _r(ak, bk, Wtt).T)  # R W_tt R^T, (sin, cos) indexed
                RW_te = _r(ak, bk, W[:2 * K, ei])  # R W_te: (2K, n_e)
                Ru = _r(ak, bk, ut[:, None])[:, 0]
                Wee = W[np.ix_(ei, ei)]
                ue = u[ei]
                g = (-0.5 * np.einsum("ij,rij->r", RWRt, blk["G"])
                     - np.einsum("ij,rij->r", RW_te, blk["GE"])
                     - 0.5 * np.einsum("ij,rij->r", Wee, blk["EE"])
                     + blk["h"] @ Ru + blk["hE"] @ ue - 0.5 * blk["yy"])
                sig2 = pt["sig2"]
                ds[k] = g @ pt["dq_ds"] - np.sum(s[k] * blk["e2"] / sig2)
                dj[k] = g @ pt["dq_dj"] - np.sum(j[k] / sig2)
        return L, da, db, ds, dj

    def conditional(self, a, b, s, j):
        """Conditional mean of every linear parameter and the lower Cholesky
        factor of its precision ``P``: draws are ``mean + L^-T eps``."""
        self.build()
        a, b, s, j = (np.asarray(x, dtype=np.float64) for x in (a, b, s, j))
        P, cvec, _, _, _ = self._assemble(a, b, s, j)
        cf = _cholesky(P)
        if cf is None:
            nan = np.full(self.p, np.nan)
            return nan, np.full((self.p, self.p), np.nan)
        Lc = np.tril(cf[0])
        mean = sla.cho_solve((Lc, True), cvec, check_finite=False)
        return mean, Lc

    def precision(self, a, b, s, j):
        """The linear parameters' conditional precision ``P`` (float64)."""
        self.build()
        a, b, s, j = (np.asarray(x, dtype=np.float64) for x in (a, b, s, j))
        return self._assemble(a, b, s, j)[0]

    # ------------------------------------------------------------ JAX
    def loglik_fn(self):
        """JAX function ``(a, b, s, j) -> log likelihood`` (float32), with the
        exact gradient; batched calls run one at a time on the host."""
        if self._jax_fns is not None:
            return self._jax_fns[0]
        import jax
        import jax.numpy as jnp

        def cb(fn, shapes, *args, batched=False):
            # batched: the host function takes leading batch dimensions itself
            # (one call per vmapped batch, e.g. the nested solve's draws).
            try:
                return jax.pure_callback(fn, shapes, *args,
                                         vmap_method="broadcast_all" if batched else "sequential")
            except TypeError:  # jax < 0.4.34
                return jax.pure_callback(fn, shapes, *args, vectorized=batched)

        def value_host(a, b, s, j):
            self.build()
            a, b, s, j = (np.asarray(x, dtype=np.float64) for x in (a, b, s, j))
            if a.ndim == 2:
                return np.asarray(self._value(a, b, s, j), dtype=np.float32)
            batch = a.shape[:-2]
            a, b = a.reshape((-1,) + a.shape[-2:]), b.reshape((-1,) + b.shape[-2:])
            s, j = s.reshape((-1, s.shape[-1])), j.reshape((-1, j.shape[-1]))
            out = np.array([self._value(*x) for x in zip(a, b, s, j)], dtype=np.float32)
            return out.reshape(batch)

        def vg_host(a, b, s, j):
            L, da, db, ds, dj = self.value_and_grad(a, b, s, j)
            return (np.asarray(L, dtype=np.float32), da.astype(np.float32), db.astype(np.float32),
                    ds.astype(np.float32), dj.astype(np.float32))

        @jax.custom_vjp
        def loglik(a, b, s, j):
            return cb(value_host, jax.ShapeDtypeStruct((), jnp.float32), a, b, s, j, batched=True)

        def fwd(a, b, s, j):
            shapes = (jax.ShapeDtypeStruct((), jnp.float32), jax.ShapeDtypeStruct(a.shape, jnp.float32),
                      jax.ShapeDtypeStruct(b.shape, jnp.float32), jax.ShapeDtypeStruct(s.shape, jnp.float32),
                      jax.ShapeDtypeStruct(j.shape, jnp.float32))
            L, da, db, ds, dj = cb(vg_host, shapes, a, b, s, j)
            return L, (da, db, ds, dj)

        def bwd(res, g):
            return tuple(g * r for r in res)

        loglik.defvjp(fwd, bwd)

        def conditional_host(a, b, s, j):
            mean, Lc = self.conditional(a, b, s, j)
            return mean.astype(np.float32), Lc.astype(np.float32)

        def conditional(a, b, s, j):
            shapes = (jax.ShapeDtypeStruct((self.p,), jnp.float32),
                      jax.ShapeDtypeStruct((self.p, self.p), jnp.float32))
            return cb(conditional_host, shapes, a, b, s, j)

        self._jax_fns = (loglik, conditional)
        return loglik

    def conditional_fn(self):
        self.loglik_fn()
        return self._jax_fns[1]


def _cholesky(P):
    """``cho_factor`` of ``P``, or None if it is not numerically positive
    definite or not finite."""
    if not np.all(np.isfinite(P)):
        return None
    try:
        return sla.cho_factor(P, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None


def _r(a, b, X):
    """R X for X indexed by (S, C) rows: rows of the result are (sin, cos)."""
    K = len(a)
    XS, XC = X[:K], X[K:]
    return np.vstack([a[:, None] * XS + b[:, None] * XC, -b[:, None] * XS + a[:, None] * XC])


def _rt(a, b, X):
    """R^T X for X indexed by (sin, cos) rows: rows of the result are (S, C)."""
    K = len(a)
    Xs, Xc = X[:K], X[K:]
    return np.vstack([a[:, None] * Xs - b[:, None] * Xc, b[:, None] * Xs + a[:, None] * Xc])
