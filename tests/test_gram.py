"""The Gram-matrix marginal likelihood (discotau.gram) against the QR of
model._marginal_solve: same log likelihood, same gradient and the same
conditional mean, for every kind of light curve and the error model. Quick:
the model is evaluated, never fitted."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

import discotau.model as model
from discotau import EchoFit
from discotau.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="module")
def data():
    return generate_synthetic_dataset(bands={"g": 4770.0, "i": 7625.0, "u": 3540.0}, M_BH=1.0e8,
                                      n_obs_per_band=40, n_freq=12, n_tau=60, seed=3)


def _echofit(data, driver=False, error_model=(), background_order=1, dce=()):
    ef = EchoFit(M_BH=1.0e8, marginalise_linear=True)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"],
                          fit_error_model=name in error_model, background_order=background_order,
                          diffuse_continuum=name in dce)
    if driver:
        d = data["bands"]["u"]
        ef.add_driver_lightcurve(t=d["t"], y=d["y"] * 1.3 + 0.1, yerr=d["yerr"], fit_error_model="driver" in error_model)
    ef.build_grid(n_freq=12, n_tau=60)
    return ef


def _params(ef, rng=0):
    """A random point in the nonlinear parameters (the model's own sites)."""
    kw = ef._model_kwargs()
    with handlers.seed(rng_seed=rng):
        tr = handlers.trace(model.reverberation_model).get_trace(**kw)
    return {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed", False)
            and k != "linear_eps"}


def _logp(ef, params, method, monkeypatch):
    monkeypatch.setattr(model, "MARGINAL_METHOD", method)
    kw = ef._model_kwargs()
    return log_density(model.reverberation_model, (), kw, params)[0]


CASES = {
    "plain": dict(),
    "driver": dict(driver=True),
    "error model": dict(error_model=("g", "i")),
    "error model and driver": dict(driver=True, error_model=("u", "driver")),
    "no background": dict(background_order=0),
    "order-2 background, diffuse continuum": dict(background_order=2, dce=("u",)),
}


@pytest.fixture(params=["auto", "dense", "gram"])
def host(request, monkeypatch):
    import discotau.gram as gram

    monkeypatch.setattr(gram, "HOST_METHOD", request.param)
    return request.param


@pytest.mark.parametrize("case", list(CASES))
def test_gram_log_likelihood_matches_the_qr(data, case, host, monkeypatch):
    ef = _echofit(data, **CASES[case])
    for rng in (0, 1):
        params = _params(ef, rng)
        if "sigma_jitter_g" in params:
            params["sigma_jitter_g"] = jnp.asarray(0.3 * float(np.mean(data["bands"]["g"]["yerr"])))
        qr = float(_logp(ef, params, "qr", monkeypatch))
        gram = float(_logp(ef, params, "gram", monkeypatch))
        # The QR is float32: ~1e-5 relative (0.06 at |L| ~ 9000, checked against a
        # float64 evaluation, which the Gram path matches to the last digit).
        assert gram == pytest.approx(qr, abs=2e-3 + 2e-5 * abs(qr))


@pytest.mark.parametrize("case", ["plain", "error model and driver"])
def test_gram_gradient_matches_the_qr(data, case, host, monkeypatch):
    ef = _echofit(data, **CASES[case])
    params = _params(ef, 2)
    grads = {}
    for method in ("qr", "gram"):
        monkeypatch.setattr(model, "MARGINAL_METHOD", method)
        kw = ef._model_kwargs()
        grads[method] = jax.grad(lambda p: log_density(model.reverberation_model, (), kw, p)[0])(params)
    for k in params:
        g_qr, g_gram = np.asarray(grads["qr"][k]), np.asarray(grads["gram"][k])
        scale = max(1.0, float(np.max(np.abs(g_qr))))
        assert np.allclose(g_gram, g_qr, atol=5e-3 * scale, rtol=5e-3), k


def test_gram_conditional_mean_predictions_match_the_qr(data, host, monkeypatch):
    # With linear_eps pinned to zero the draw is the conditional mean, the same
    # in both factorisations (their eps maps differ but are equal in distribution).
    ef = _echofit(data, driver=True, error_model=("g",))
    params = _params(ef, 4)
    preds = {}
    for method in ("qr", "gram"):
        monkeypatch.setattr(model, "MARGINAL_METHOD", method)
        kw = dict(ef._model_kwargs(), draw_linear=True)
        n_lin = 2 * len(ef.freqs) + 4 + 4  # S, C, four offsets, four linear backgrounds
        sub = dict(params, linear_eps=jnp.zeros(n_lin))
        with handlers.seed(rng_seed=0), handlers.substitute(data=sub):
            tr = handlers.trace(model.reverberation_model).get_trace(**kw)
        preds[method] = {k: np.asarray(v["value"]) for k, v in tr.items() if k.startswith(("y_pred_", "C_", "bg_"))}
    for k, v in preds["qr"].items():
        assert np.allclose(preds["gram"][k], v, rtol=2e-3, atol=2e-3 * float(np.std(v) + 1e-12)), k


def test_gram_path_forms_no_data_by_mode_columns(data, monkeypatch):
    # The potential must not depend on N x K products: _marginal_solve is not called.
    ef = _echofit(data)
    params = _params(ef, 5)
    calls = []
    monkeypatch.setattr(model, "_marginal_solve", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(model, "MARGINAL_METHOD", "gram")
    log_density(model.reverberation_model, (), ef._model_kwargs(), params)
    assert not calls


def test_error_model_weight_family_is_low_rank_and_exact(data):
    from discotau.gram import GramCache

    d = data["bands"]["g"]
    yerr = np.asarray(d["yerr"]) * np.linspace(0.6, 2.0, len(d["yerr"]))
    cache = GramCache(np.linspace(0.1, 2.0, 5), {"g": dict(t=d["t"], y=d["y"], yerr=yerr, error_model=True,
                                                           offset_loc=0.0, offset_sd=1.0, bg_cols=None,
                                                           bg_sd=None)}, method="gram").build()
    blk = cache._blocks[0]
    assert 3 <= blk["U"].shape[1] <= 20
    e2 = yerr ** 2
    for s, j in [(1.0, 0.0), (0.7, 0.5 * float(np.mean(yerr))), (1.6, 3.0 * float(np.mean(yerr)))]:
        q, _, _ = cache._coeffs(blk, s, j)
        w_exact = 1.0 / (s ** 2 * e2 + j ** 2)
        assert np.allclose(blk["U"] @ q, w_exact, rtol=1e-8)


def _fake_optimum(ef, rng=6):
    """A point and an identity Laplace covariance standing in for optimise()'s
    (the quick tier never solves)."""
    from jax.flatten_util import ravel_pytree
    from numpyro.infer import init_to_value
    from discotau.echofit import _initialize_marginal_model

    params = _params(ef, rng)
    marg = dict(ef._model_kwargs(), marginalise_linear=True)
    z = _initialize_marginal_model(0, marg, init_to_value(values=params), ef.M_BH).param_info.z
    ef.optimum = {k: np.asarray(v) for k, v in params.items()}
    ef.laplace_covariance = np.eye(ravel_pytree(z)[0].size)
    return params


@pytest.mark.parametrize("marginalise", [False, True])
def test_optimum_inverse_mass_covers_every_site_and_is_positive_definite(data, marginalise):
    ef = _echofit(data, driver=True, error_model=("g",))
    ef.marginalise_linear = marginalise
    _fake_optimum(ef)
    (names, M), = ef._optimum_inverse_mass().items()
    M = np.asarray(M, dtype=np.float64)
    assert list(names) == sorted(names)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0
    assert ("S_raw" in names) == (not marginalise)


def test_optimum_inverse_mass_linear_block_is_the_conditional_covariance(data):
    # The S_raw/C_raw block is P^-1 from the Gram cache (unit-normal sites, no rescaling).
    import discotau.model as m
    from numpyro import handlers

    ef = _echofit(data)
    ef.marginalise_linear = False
    params = _fake_optimum(ef)
    (names, M), = ef._optimum_inverse_mass().items()
    marg = dict(ef._model_kwargs(), marginalise_linear=True)
    grabbed = {}
    original = m._linear_marginal

    def grab(blocks, prior_scale, draw_linear, gram_cache=None):
        grabbed["x"] = m._gram_inputs(blocks, prior_scale)
        return original(blocks, prior_scale, draw_linear, gram_cache)

    m._linear_marginal = grab
    try:
        handlers.trace(handlers.substitute(handlers.seed(m.reverberation_model, 0), params)).get_trace(**marg)
    finally:
        m._linear_marginal = original
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    K = len(ef.freqs)
    Pinv = np.linalg.inv(marg["gram_cache"].precision(*grabbed["x"]))
    z = initialize_model(jax.random.PRNGKey(0), m.reverberation_model, model_kwargs=ef._model_kwargs(),
                         init_strategy=init_to_value(values=ef._optimum_init_values()), validate_grad=False).param_info.z
    start = sum(int(np.size(z[n])) for n in names[:list(names).index("C_raw")])
    block = np.asarray(M, dtype=np.float64)[start:start + K, start:start + K]
    assert np.allclose(block, Pinv[K:2 * K, K:2 * K], rtol=1e-3, atol=1e-6 * np.abs(Pinv).max())


@pytest.mark.fit
def test_nuts_started_from_the_direct_solve_runs_and_agrees(data):
    ef = _echofit(data)
    ef.optimise(method="laplace", num_restarts=1, num_samples=100)
    lm_opt = float(np.mean(ef.samples["log_mdot_msun_yr"]))
    sd_opt = float(np.std(ef.samples["log_mdot_msun_yr"]))
    ef.fit(num_warmup=150, num_samples=150, init_from_optimum=True, mass_from_optimum=True,
           progress_bar=False, generate_report=False)
    lm = np.asarray(ef.samples["log_mdot_msun_yr"])
    assert abs(float(lm.mean()) - lm_opt) < 3 * max(sd_opt, float(lm.std()))
    assert int(np.asarray(ef.extra_fields["diverging"]).sum()) < 15


def test_auto_uses_gram_only_with_enough_points_per_linear_parameter(data, monkeypatch):
    # 3 x 40 points against 2 x 12 modes + 3 offsets + 3 backgrounds = 30 linear
    # parameters: 4 points each.
    ef = _echofit(data)
    params = _params(ef, 7)
    monkeypatch.setattr(model, "MARGINAL_METHOD", "auto")
    for ratio, expect_qr in ((1.2, False), (3.0, False), (5.0, True)):
        calls = []
        original = model._marginal_solve

        def spy(*a, **k):
            calls.append(1)
            return original(*a, **k)

        monkeypatch.setattr(model, "_marginal_solve", spy)
        monkeypatch.setattr(model, "GRAM_MIN_POINTS_PER_PARAMETER", ratio)
        log_density(model.reverberation_model, (), ef._model_kwargs(), params)
        assert bool(calls) == expect_qr
        monkeypatch.setattr(model, "_marginal_solve", original)


def _host_inputs(ef, params):
    """The Gram cache and its (a, b, s, j) at ``params``, from the model itself."""
    from numpyro import handlers

    marg = dict(ef._model_kwargs(), marginalise_linear=True)
    grabbed = {}
    original = model._linear_marginal

    def grab(blocks, prior_scale, draw_linear, gram_cache=None):
        grabbed["x"] = [np.asarray(v, dtype=np.float64) for v in model._gram_inputs(blocks, prior_scale)]
        return original(blocks, prior_scale, draw_linear, gram_cache)

    model._linear_marginal = grab
    try:
        handlers.trace(handlers.substitute(handlers.seed(model.reverberation_model, 0), params)).get_trace(**marg)
    finally:
        model._linear_marginal = original
    return marg["gram_cache"], grabbed["x"]


@pytest.mark.parametrize("error_model", [(), ("g", "driver")])
def test_scalar_hessian_matches_finite_differences(data, error_model, host):
    ef = _echofit(data, driver=True, error_model=error_model)
    params = _params(ef, 8)
    for n in error_model:
        params[f"sigma_jitter_{n}"] = jnp.asarray(0.4 * float(np.mean(data["bands"]["g"]["yerr"])))
    cache, (a, b, s, j) = _host_inputs(ef, params)
    H, names = cache.scalar_hessian(a, b, s, j)

    def grad(x):
        # x: the log offsets of every parameter in ``names``
        aa, bb, ss, jj = a.copy(), b.copy(), s.copy(), j.copy()
        for v, (kind, k) in zip(x, names):
            if kind == "lambda":
                aa[k] *= np.exp(v)
                bb[k] *= np.exp(v)
            elif kind == "log_s":
                ss[k] *= np.exp(v)
            else:
                jj[k] *= np.exp(v)
        _, da, db, ds, dj = cache.value_and_grad(aa, bb, ss, jj)
        out = []
        for kind, k in names:
            if kind == "lambda":
                out.append(float(da[k] @ aa[k] + db[k] @ bb[k]))
            elif kind == "log_s":
                out.append(float(ds[k] * ss[k]))
            else:
                out.append(float(dj[k] * jj[k]))
        return np.array(out)

    n = len(names)
    eps = 1e-4
    fd = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        fd[:, i] = (grad(e) - grad(-e)) / (2 * eps)
    fd = 0.5 * (fd + fd.T)
    assert np.allclose(H, fd, rtol=2e-4, atol=2e-4 * np.abs(fd).max()), np.max(np.abs(H - fd))


@pytest.mark.parametrize("error_model", [(), ("g", "driver")])
def test_exact_potential_hessian_matches_finite_differences(data, error_model, monkeypatch):
    from jax.flatten_util import ravel_pytree
    from numpyro.infer import init_to_value

    from discotau.echofit import _initialize_marginal_model
    from discotau.exact_hessian import ExactHessian
    from discotau.nested_laplace import fd_hessian

    monkeypatch.setattr(model, "MARGINAL_METHOD", "gram")
    ef = _echofit(data, driver=True, error_model=error_model)
    params = _params(ef, 9)
    for n in error_model:
        params[f"sigma_jitter_{n}"] = jnp.asarray(0.4 * float(np.mean(data["bands"]["g"]["yerr"])))
    kw = dict(ef._model_kwargs(), marginalise_linear=True)
    info = _initialize_marginal_model(0, kw, init_to_value(values=params), ef.M_BH)
    z0, unravel = ravel_pytree(info.param_info.z)
    grad = jax.jit(jax.grad(lambda z: info.potential_fn(unravel(z))))
    full_grad = lambda z: np.asarray(grad(jnp.asarray(z, dtype=z0.dtype)), dtype=np.float64)  # noqa: E731
    exact = ExactHessian.create(model.reverberation_model, kw, unravel, z0, full_grad)
    assert exact is not None
    z = np.asarray(z0, dtype=np.float64)
    H_fd, vals, vecs = fd_hessian(full_grad, z, sd=np.full(z.size, 0.05))
    sd = np.sqrt(np.sum(vecs ** 2 / np.abs(vals), axis=1))
    H = exact.hessian(z, sd)
    assert np.allclose(H, H_fd, rtol=2e-2, atol=2e-3 * np.abs(H_fd).max()), np.max(np.abs(H - H_fd))
