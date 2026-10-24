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

The complex form (light curves in Gram mode with fixed noise). With
``m_j = M_C[:, j] + i M_S[:, j] = conj(gamma_j) exp(i w_j t)``,
``gamma = a + i b``, a light curve's contribution to P's driver block is fixed
by two complex K x K matrices built once,
``E1_jk = sum_i w_i exp(i (w_j - w_k) t_i)`` (Hermitian) and
``E2_jk = sum_i w_i exp(i (w_j + w_k) t_i)`` (symmetric):

    A = diag(conj gamma) E2 diag(conj gamma),   B = diag(conj gamma) E1 diag(gamma),
    P_CC = Re(A + B) / 2,  P_SS = Re(B - A) / 2,  P_CS = Im(A - B) / 2  (C rows, S columns).

Summed over light curves by ``einsum("bj,bk,bjk->jk")``, one streaming pass
each with no K x K temporaries. With ``dL/dP = -W / 2`` and
``X = (G_CC - G_SS)/2 - i G_CS``, ``Y = (G_CC + G_SS)/2 + i G_CS`` (G the
blocks of ``-W/2``), ``dL = Re sum(X o dA + Y o dB)``, so
``dL/da = Re(g1 + g2)``, ``dL/db = Im(g1 - g2)`` with
``g1 = ((X + X^T) o E2) conj(gamma) + (Y o E1) gamma`` and
``g2_k = sum_j Y_jk E1_jk conj(gamma_j)``: three more passes. On Kaggle this
made the gradient 1.5-2.6x faster (NGC 5548's cadence, 768 modes, 1.1 s ->
0.43 s), the assembly ~4.5x; the same numbers to ~1e-11. Light curves with
the error model stay in dense mode ("auto"). Threading the einsums over light
curves gave nothing: they are memory-bound.

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

# Weight-family basis (Gram mode with the error model): singular values kept
# down to this fraction of the largest, over u from U_SPAN[0] to U_SPAN[1] times
# the mean squared error (plus u = 0).
WEIGHT_RANK_TOL = 1e-10
U_SPAN = (1e-6, 1e4)
# How each light curve enters P. "gram": a sum of its precomputed Gram matrices
# (memory-bound passes over (2K)^2 values; r ~ 10 times more with the error
# model). "dense": its rows of the weighted design matrix, formed each call from
# the stored basis, into one BLAS-3 syrk (compute-bound, the error model exact
# at no extra cost). "auto": Gram for light curves whose noise is fixed, dense
# for those with the error model. Per gradient on Kaggle (seconds, QR / Gram /
# dense): Fairall 9's cadence 0.66 / 0.23 / 0.42; NGC 5548's 0.65 / 0.30 / 0.48,
# and with the error model 0.64 / 0.80 / 0.48.
HOST_METHOD = "auto"


class GramCache:
    """Precomputed pieces of every light curve for one frequency grid, built
    on first use. ``curves``: block name -> dict with ``t``, ``y``, ``yerr``,
    ``error_model`` (bool), ``offset_loc``, ``offset_sd``, ``bg_cols``
    (``(n, K)`` or None) and ``bg_sd``, in the order ``model._linear_marginal``
    stacks them (the driver light curve first). ``method`` overrides
    ``HOST_METHOD`` for this cache."""

    def __init__(self, freqs, curves: dict, method=None):
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.curves = curves
        self.names = list(curves)
        self.method = method or HOST_METHOD
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

    def _mode(self, error_model):
        if self.method == "auto":
            return "dense" if error_model else "gram"
        return self.method

    def build(self):
        if self._blocks is not None:
            return self
        w, K = self.freqs, self.K
        blocks, bg0, row0 = [], 2 * K + self.n_off, 0
        for i, (name, c) in enumerate(self.curves.items()):
            t = np.asarray(c["t"], dtype=np.float64)
            y = np.asarray(c["y"], dtype=np.float64)
            e2 = np.asarray(c["yerr"], dtype=np.float64) ** 2
            yt = y - float(c["offset_loc"])
            cols = [np.full(len(t), float(c["offset_sd"]))]
            idx = [2 * K + i]
            if c["bg_cols"] is not None:
                kb = c["bg_cols"].shape[1]
                cols += list((np.asarray(c["bg_cols"], dtype=np.float64) * float(c["bg_sd"])).T)
                idx += list(range(bg0, bg0 + kb))
                bg0 += kb
            E = np.column_stack(cols)
            wt = np.outer(t, w)
            mode = self._mode(bool(c["error_model"]))
            blk = dict(name=name, e2=e2, e_idx=np.array(idx), mode=mode, projected=bool(c["error_model"]),
                       sin=np.sin(wt), cos=np.cos(wt), E=E, yt=yt)
            if mode == "dense":
                blk.update(rows=slice(row0, row0 + len(t)))
                row0 += len(t)
            else:
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
                blk.update(U=U, projected=projected, Gss=np.ascontiguousarray(G[:, :K, :K]),
                           Gsc=np.ascontiguousarray(G[:, :K, K:]), Gcc=np.ascontiguousarray(G[:, K:, K:]),
                           G=G, GE=np.stack(GE), EE=np.stack(EE), h=np.stack(h), hE=np.stack(hE), yy=np.array(yy))
            blocks.append(blk)
        # Gram-mode light curves with fixed noise go to the complex form: E1 and
        # E2 replace their Gss, Gsc, Gcc (the complex form, module docstring).
        self._cx = np.array([k for k, b in enumerate(blocks) if b["mode"] == "gram" and not b["projected"]], int)
        if len(self._cx):
            E1, E2 = [], []
            for k in self._cx:
                blk = blocks[k]
                blk["mode"] = "complex"
                Gss, Gsc, Gcc = blk.pop("Gss")[0], blk.pop("Gsc")[0], blk.pop("Gcc")[0]
                del blk["G"]
                E1.append((Gcc + Gss) + 1j * (Gsc - Gsc.T))
                E2.append((Gcc - Gss) + 1j * (Gsc + Gsc.T))
            self._E1, self._E2 = np.stack(E1), np.stack(E2)
        self._blocks = blocks
        self.n_dense = row0
        self.N = int(sum(len(b["e2"]) for b in blocks))
        return self

    # ------------------------------------------------------------ numerics
    def _coeffs(self, blk, s, j):
        """Gram mode: weight coefficients q_r (the Gram quantities are
        sum_r q_r X_r) and their derivatives with respect to s and j."""
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
        projections c, y^T D^-1 y, log det D, and per-block pieces the gradient
        reuses. Gram blocks: R^T G R accumulated into P's four K x K blocks in
        place (with G R's blocks GR_sS = G_ss a - G_sc b, GR_sC = G_ss b + G_sc a
        and cos rows alike: P_SS = a GR_sS - b GR_cS, P_SC = a GR_sC - b GR_cC,
        P_CS = b GR_sS + a GR_cS, P_CC = b GR_sC + a GR_cC). Dense blocks: their
        rows of M~ = D^-1/2 [Phi R | E], then one syrk for all of them."""
        K, p = self.K, self.p
        P = np.zeros((p, p))
        cvec = np.zeros(p)
        yy = logdetD = 0.0
        parts = []
        PSS, PSC, PCS, PCC = P[:K, :K], P[:K, K:2 * K], P[K:2 * K, :K], P[K:2 * K, K:2 * K]
        tmp = np.empty((K, K))
        Mt = np.zeros((self.n_dense, p), order="F") if self.n_dense else None
        yw = np.empty(self.n_dense)
        for k, blk in enumerate(self._blocks):
            ak, bk = a[k], b[k]
            sig2 = s[k] ** 2 * blk["e2"] + j[k] ** 2
            logdetD += float(np.sum(np.log(sig2)))
            if blk["mode"] == "dense":
                sw = 1.0 / np.sqrt(sig2)
                r = blk["rows"]
                Mt[r, :K] = (blk["sin"] * ak - blk["cos"] * bk) * sw[:, None]
                Mt[r, K:2 * K] = (blk["sin"] * bk + blk["cos"] * ak) * sw[:, None]
                Mt[r, blk["e_idx"]] = blk["E"] * sw[:, None]
                yw[r] = blk["yt"] * sw
                parts.append(dict(sw=sw, sig2=sig2))
                continue
            q, dq_ds, dq_dj = self._coeffs(blk, s[k], j[k])
            GE = np.tensordot(q, blk["GE"], 1)
            if blk["mode"] == "complex":
                # P_tt for these comes from the einsums after the loop.
                RtGE = _rt(ak, bk, GE)
                ei = blk["e_idx"]
                P[:2 * K, ei] += RtGE
                P[ei, :2 * K] += RtGE.T
                P[np.ix_(ei, ei)] += np.tensordot(q, blk["EE"], 1)
                h = q @ blk["h"]
                cvec[:2 * K] += _rt(ak, bk, h[:, None])[:, 0]
                cvec[ei] += q @ blk["hE"]
                yy += float(q @ blk["yy"])
                parts.append(dict(GE=GE, h=h, sig2=sig2))
                continue
            if blk["projected"]:
                Gss, Gsc, Gcc = (np.tensordot(q, blk[n], 1) for n in ("Gss", "Gsc", "Gcc"))
            else:
                Gss, Gsc, Gcc = blk["Gss"][0], blk["Gsc"][0], blk["Gcc"][0]
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
            parts.append(dict(q=q, dq_ds=dq_ds, dq_dj=dq_dj, GR=(GR_sS, GR_sC, GR_cS, GR_cC), GE=GE, h=h,
                              sig2=sig2))
        if len(self._cx):
            cg = a[self._cx] - 1j * b[self._cx]
            A = np.einsum("bj,bk,bjk->jk", cg, cg, self._E2)
            Bm = np.einsum("bj,bk,bjk->jk", cg, np.conj(cg), self._E1)
            P[:K, :K] += 0.5 * (Bm.real - A.real)
            P[K:2 * K, K:2 * K] += 0.5 * (A.real + Bm.real)
            PCS = 0.5 * (A.imag - Bm.imag)
            P[K:2 * K, :K] += PCS
            P[:K, K:2 * K] += PCS.T
        if self.n_dense:
            from scipy.linalg.blas import dsyrk

            Pd = dsyrk(1.0, Mt, trans=1, lower=0)
            P += np.triu(Pd) + np.triu(Pd, 1).T
            cvec += Mt.T @ yw
            yy += float(yw @ yw)
        P[np.diag_indices(p)] += 1.0
        return P, cvec, yy, logdetD, (parts, Mt, yw)

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
        P, cvec, yy, logdetD, (parts, Mt, yw) = self._assemble(a, b, s, j)
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
        W = _cho_inverse(cf)
        W += np.outer(u, u)
        Wtt, ut = W[:2 * K, :2 * K], u[:2 * K]
        WS, WC = Wtt[:K], Wtt[K:]
        da, db = np.zeros_like(a), np.zeros_like(b)
        ds, dj = np.zeros_like(s), np.zeros_like(j)
        if len(self._cx):
            # dL/dP = -W/2; on the (S, C) block, dL = Re sum(X o dA + Y o dB) (the complex form, module docstring).
            GSS, GCC, GCS = -0.5 * W[:K, :K], -0.5 * W[K:2 * K, K:2 * K], -0.5 * W[K:2 * K, :K]
            X = 0.5 * (GCC - GSS) - 1j * GCS
            Y = 0.5 * (GCC + GSS) + 1j * GCS
            cg = a[self._cx] - 1j * b[self._cx]
            g1 = (np.einsum("jk,bjk,bk->bj", X + X.T, self._E2, cg)
                  + np.einsum("jk,bjk,bk->bj", Y, self._E1, np.conj(cg)))
            # E1 is Hermitian: sum_j Y_jk E1_jk conj(gamma_j) = conj(sum_j conj(Y)_jk E1_kj gamma_j).
            g2 = np.conj(np.einsum("kj,bkj,bj->bk", np.ascontiguousarray(np.conj(Y.T)), self._E1, np.conj(cg)))
            da[self._cx] = (g1 + g2).real
            db[self._cx] = (g1 - g2).imag
        if self.n_dense:
            dM = np.outer(yw, u) - Mt @ W          # dL/dM~ for the dense rows
            dy = Mt @ u - yw                        # dL/dy~
        for k, (blk, pt) in enumerate(zip(self._blocks, parts)):
            ei = blk["e_idx"]
            if blk["mode"] == "dense":
                r, sw = blk["rows"], pt["sw"]
                gS = dM[r, :K] * sw[:, None]
                gC = dM[r, K:2 * K] * sw[:, None]
                sn, cs = blk["sin"], blk["cos"]
                # M~_S = sw (sin a - cos b), M~_C = sw (sin b + cos a).
                da[k] = np.sum(gS * sn + gC * cs, axis=0)
                db[k] = np.sum(-gS * cs + gC * sn, axis=0)
                if blk["projected"]:
                    cols = np.r_[0:2 * K, ei]
                    # Every weighted quantity in a row is proportional to its sw.
                    dsw = (np.sum(dM[r][:, cols] * Mt[r][:, cols], axis=1) + dy[r] * yw[r]) / sw
                    dsig2 = dsw * (-0.5) * sw ** 3 - 0.5 / pt["sig2"]
                    ds[k] = float(np.sum(dsig2 * 2.0 * s[k] * blk["e2"]))
                    dj[k] = float(np.sum(dsig2 * 2.0 * j[k]))
                continue
            if blk["mode"] == "complex":
                # The data projections (dL/dc = u) and the offset/background cross block
                # (dL/dP_tE = -W_tE, both copies), with R^T rows S = a X_s - b X_c, C = b X_s + a X_c.
                h, GE = pt["h"], pt["GE"]
                WSE, WCE = W[:K, ei], W[K:2 * K, ei]
                GEs, GEc = GE[:K], GE[K:]
                da[k] += ut[:K] * h[:K] + ut[K:] * h[K:] - np.sum(WSE * GEs + WCE * GEc, axis=1)
                db[k] += -ut[:K] * h[K:] + ut[K:] * h[:K] - np.sum(WCE * GEs - WSE * GEc, axis=1)
                continue
            Wet = W[ei, :2 * K]
            (GR_sS, GR_sC, GR_cS, GR_cC), GE, h = pt["GR"], pt["GE"], pt["h"]
            # dL/dR = h u^T - G R W_tt - G_E W_et: diagonals of its four K x K
            # sub-blocks; (G R W)_kk = GR_rS[k] . W_S,c[:, k] + GR_rC[k] . W_C,c[:, k].

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
                g = (-0.5 * np.einsum("ij,rij->r", RWRt, blk["G"])
                     - np.einsum("ij,rij->r", RW_te, blk["GE"])
                     - 0.5 * np.einsum("ij,rij->r", Wee, blk["EE"])
                     + blk["h"] @ Ru + blk["hE"] @ u[ei] - 0.5 * blk["yy"])
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
            return np.full(self.p, np.nan), np.full((self.p, self.p), np.nan)
        Lc = np.tril(cf[0])
        return sla.cho_solve((Lc, True), cvec, check_finite=False), Lc

    def precision(self, a, b, s, j):
        """The linear parameters' conditional precision ``P`` (float64)."""
        self.build()
        a, b, s, j = (np.asarray(x, dtype=np.float64) for x in (a, b, s, j))
        return self._assemble(a, b, s, j)[0]

    def scalar_hessian(self, a, b, s, j):
        """Exact Hessian of the log likelihood with respect to every light
        curve's log scale ``lambda_b`` (``a, b -> e^dlambda (a, b)``: the log of
        gain_b * sigma_drw) and, for light curves with the error model, the logs
        of its error scale and jitter. Returns ``(H, names)`` with ``names`` the
        ``("lambda" | "log_s" | "log_j", block index)`` of each row.

        With K = D + Z Z^T, Q = K^-1, alpha = Q y:
        d2L = 1/2 a^T K_tt' a - a^T K_t Q K_t' a - 1/2 tr(Q K_tt') + 1/2 tr(Q K_t Q K_t'),
        moved into p x p quantities through Z^T Q = S Z^T W, Q Z = W Z S,
        Z^T a = u and Z^T Q Z = I - S (S = P^-1); only the noise-noise term
        sum_ik Q_ik^2 D_t,i D_t',k needs Q itself, formed once (O(N^2 p))."""
        self.build()
        a, b, s, j = (np.asarray(x, dtype=np.float64) for x in (a, b, s, j))
        K, p = self.K, self.p
        blocks = self._blocks
        sizes = [len(blk["e2"]) for blk in blocks]
        starts = np.cumsum([0] + sizes)
        N = int(starts[-1])
        Z = np.zeros((N, p))
        Zth = []
        Dvec = np.empty(N)
        y = np.empty(N)
        for k, blk in enumerate(blocks):
            r = slice(starts[k], starts[k + 1])
            Z[r, :K] = blk["sin"] * a[k] - blk["cos"] * b[k]
            Z[r, K:2 * K] = blk["sin"] * b[k] + blk["cos"] * a[k]
            Z[r, blk["e_idx"]] = blk["E"]
            Dvec[r] = s[k] ** 2 * blk["e2"] + j[k] ** 2
            y[r] = blk["yt"]
        w = 1.0 / Dvec
        WZ = Z * w[:, None]
        P = np.eye(p) + Z.T @ WZ
        cf = sla.cho_factor(P, lower=True, check_finite=False)
        Sg = _cho_inverse(cf)
        u = Sg @ (WZ.T @ y)
        alpha = w * (y - Z @ u)
        V = WZ @ Sg                                   # Q Z = W Z S
        Qdiag = w - np.einsum("ij,ij->i", V, WZ)      # diag(W - W Z S Z^T W)
        # parameters: (kind, block, D_t, D_tt) with D_t, D_tt per-row vectors on the block
        params = []
        for k, blk in enumerate(blocks):
            params.append(("lambda", k))
        for k, blk in enumerate(blocks):
            if blk["projected"]:
                params.append(("log_s", k))
                params.append(("log_j", k))
        n = len(params)
        rows_of = [slice(starts[k], starts[k + 1]) for k in range(len(blocks))]
        # per-parameter pieces
        info = []
        for kind, k in params:
            r = rows_of[k]
            if kind == "lambda":
                Zt = np.zeros((sizes[k], p))
                Zt[:, :2 * K] = Z[r, :2 * K]
                Bt = WZ[r].T @ Zt                     # Z^T W Z_t (only block rows)
                g = Zt.T @ alpha[r]                   # Z_t^T alpha
                C = Sg @ Bt                           # S B_t
                beta = np.zeros(N)
                beta[r] += Zt @ u
                beta += Z @ g
                info.append(dict(kind=kind, k=k, Zt=Zt, B=Bt, g=g, C=C, F=C @ Sg, beta=beta, d1=None, v=g - Bt @ u))
            else:
                e2 = blocks[k]["e2"]
                d1 = 2.0 * (s[k] ** 2 * e2 if kind == "log_s" else np.full(sizes[k], j[k] ** 2))
                X = (Z[r] * (w[r] ** 2 * d1)[:, None]).T @ Z[r]   # Z^T W D_t W Z
                beta = np.zeros(N)
                beta[r] = d1 * alpha[r]
                info.append(dict(kind=kind, k=k, d1=d1, d2=2.0 * d1, SX=Sg @ X, beta=beta))
        noise = [i for i, it in enumerate(info) if it["kind"] != "lambda"]
        if noise:
            Q = np.diag(w) - V @ WZ.T                 # K^-1, formed once
            Dm = np.zeros((N, len(noise)))
            for c, i in enumerate(noise):
                Dm[rows_of[info[i]["k"]], c] = info[i]["d1"]
            QQD = (Q * Q) @ Dm                        # sum_k Q_ik^2 D_t',k
            term_a = Dm.T @ QQD
        ZWb = [Z.T @ (w * it["beta"]) for it in info]  # Z^T W beta_t
        H = np.zeros((n, n))
        for i_, ti in enumerate(info):
            for i2 in range(i_, n):
                tj = info[i2]
                same = i_ == i2
                if not (ti["kind"] == "lambda" and tj["kind"] == "lambda"):
                    # T2 = -beta_t^T Q beta_t'
                    h = -(ti["beta"] @ (w * tj["beta"]) - ZWb[i_] @ Sg @ ZWb[i2])
                if ti["kind"] == "lambda" and tj["kind"] == "lambda":
                    # T1 + T2 together: separately each is ~1e3 times the sum
                    # (5e7 against 2e4 on a small test set) and the p-space T2
                    # lost ~3e-6 of it in float64. With Q beta_t' =
                    # W Z_t' u + W Z S (g_t' - B_t' u), T1 + T2 =
                    # [t = t'] g_t . u - u^T Z_t'^T W Z_t u + v_t^T S v_t', v = g - B u.
                    h = ti["v"] @ Sg @ tj["v"]
                    if ti["k"] == tj["k"]:
                        r = rows_of[ti["k"]]
                        Zu = ti["Zt"] @ u
                        h -= Zu @ (w[r] * Zu)
                    t3 = -np.sum(tj["B"] * ti["C"])                            # + tr(B_t'^T S B_t) inside -[...]
                    if ti["k"] == tj["k"]:
                        r = rows_of[ti["k"]]
                        Gbb = ti["Zt"].T @ (ti["Zt"] * w[r][:, None])          # Z_t^T W Z_t
                        t3_same = np.trace(Gbb)
                    else:
                        Gbb, t3_same = None, 0.0
                    h += -(t3_same + t3)                                       # -tr(Z_t'^T Q Z_t)
                    if same:
                        h += ti["g"] @ u - np.trace(ti["C"])                   # T1 (Z_tt = Z_t), T3
                    d = np.sum(ti["C"] * tj["C"].T)
                    gterm = np.sum(ti["C"].T * tj["C"])
                    # (e)(t,t') + (e)(t',t): tr(Z_t'^T W Z_t (I - S)) - tr(B_t'^T S B_t (I - S)), and swapped
                    e1 = 0.0
                    if Gbb is not None:
                        e1 = 2.0 * (np.trace(Gbb) - np.sum(Gbb * Sg))
                    e2_ = -(np.sum(tj["B"] * ti["C"]) - np.sum(tj["B"] * ti["F"])) \
                          - (np.sum(ti["B"] * tj["C"]) - np.sum(ti["B"] * tj["F"]))
                    h += 0.5 * (d + gterm + e1 + e2_)
                elif ti["kind"] != "lambda" and tj["kind"] != "lambda":
                    ci, cj = noise.index(i_), noise.index(i2)
                    if same:
                        r = rows_of[ti["k"]]
                        h += 0.5 * np.sum(ti["d2"] * alpha[r] ** 2) - 0.5 * np.sum(Qdiag[r] * ti["d2"])
                    h += 0.5 * term_a[ci, cj]                                  # (a)
                else:
                    tn, tl = (ti, tj) if ti["kind"] != "lambda" else (tj, ti)
                    # (b)(noise, lambda) = tr(Q D_t Q Z_t' Z^T) = tr(S Z^T W D_t W Z_t') - tr(S X_t S B_t');
                    # T4 = 1/2 tr(Q D_t Q (Z_t' Z^T + Z Z_t'^T)) = (b), the lambda parameter having no D.
                    bterm = -np.sum(tn["SX"].T * tl["C"])
                    if tn["k"] == tl["k"]:
                        r = rows_of[tn["k"]]
                        M = (Z[r] * (w[r] ** 2 * tn["d1"])[:, None]).T @ tl["Zt"]
                        bterm += np.sum(Sg * M.T)
                    h += bterm
                H[i_, i2] = H[i2, i_] = h
        return H, params

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


def _cho_inverse(cf):
    """``P^-1`` from ``cho_factor``'s lower factor by LAPACK's potri: about a
    third fewer flops than solving against the identity (on Kaggle, factor
    plus inverse 0.24 s -> 0.11 s at p = 1574)."""
    from scipy.linalg.lapack import dpotri

    Linv, info = dpotri(cf[0], lower=1)
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotri failed (info {info})")
    return np.tril(Linv) + np.tril(Linv, -1).T


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
