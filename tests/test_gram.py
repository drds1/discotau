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


@pytest.mark.parametrize("case", list(CASES))
def test_gram_log_likelihood_matches_the_qr(data, case, monkeypatch):
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
def test_gram_gradient_matches_the_qr(data, case, monkeypatch):
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


def test_gram_conditional_mean_predictions_match_the_qr(data, monkeypatch):
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
                                                           bg_sd=None)}).build()
    blk = cache._blocks[0]
    assert 3 <= blk["U"].shape[1] <= 20
    e2 = yerr ** 2
    for s, j in [(1.0, 0.0), (0.7, 0.5 * float(np.mean(yerr))), (1.6, 3.0 * float(np.mean(yerr)))]:
        q, _, _ = cache._coeffs(blk, s, j)
        w_exact = 1.0 / (s ** 2 * e2 + j ** 2)
        assert np.allclose(blk["U"] @ q, w_exact, rtol=1e-8)
