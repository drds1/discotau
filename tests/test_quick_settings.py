"""Quick checks of what each setting puts in the model, by tracing the model
once rather than fitting it (the fitting tier checks the same settings end to
end): which sample sites exist, and the priors anchored to the data."""

import numpy as np
import pytest
from numpyro import handlers

import discotau.model as model_mod
from discotau import EchoFit
from discotau.echofit import PERIOD_MAX_FACTOR
from discotau.forward_model import disk_t1_kelvin, lag_scaling, thin_disk_response, wien_radius, _WIEN_B_ANGSTROM_KELVIN
from discotau.model import reverberation_model
from discotau.synthetic import generate_synthetic_dataset


@pytest.fixture(scope="module")
def data():
    return generate_synthetic_dataset(bands={"g": 4770.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=30,
                                      n_freq=8, n_tau=40, seed=0)


def _echofit(data, driver=False, error_model=(), **kwargs):
    ef = EchoFit(M_BH=1.0e8, **kwargs)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"],
                          fit_error_model=name in error_model)
    if driver:
        d = data["bands"]["g"]
        ef.add_driver_lightcurve(t=d["t"], y=d["y"], yerr=d["yerr"], fit_error_model="driver" in error_model)
    ef.build_grid(n_freq=8, n_tau=40)
    return ef


def _trace(ef):
    tr = handlers.trace(handlers.seed(reverberation_model, 0)).get_trace(**ef._model_kwargs())
    return tr


def _sampled(tr):
    return {k for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed", False)}


def test_error_model_sites_only_where_switched_on(data):
    assert not {s for s in _sampled(_trace(_echofit(data))) if s.startswith(("sigma_scale_", "sigma_jitter_"))}
    sites = _sampled(_trace(_echofit(data, driver=True, error_model=("g", "driver"))))
    assert {"sigma_scale_g", "sigma_jitter_g", "sigma_scale_driver", "sigma_jitter_driver"} <= sites
    assert not {"sigma_scale_i", "sigma_jitter_i"} & sites


def test_tau_drw_only_with_the_drw_prior(data):
    assert "tau_drw" not in _sampled(_trace(_echofit(data)))
    assert "tau_drw" in _sampled(_trace(_echofit(data, drw_prior=True)))


def test_inclination_is_sampled_as_cos_inclination_within_the_prior_bound(data):
    tr = _trace(_echofit(data))
    fn = tr["cos_inclination"]["fn"]
    assert float(fn.low) == pytest.approx(np.cos(np.deg2rad(model_mod.INCLINATION_MAX_DEG)))
    assert float(fn.high) == pytest.approx(1.0)
    assert tr["inclination"]["type"] == "deterministic"


def test_fixed_inclination_has_no_cos_inclination_site(data):
    tr = _trace(_echofit(data, fixed_params={"inclination": 30.0}))
    assert "cos_inclination" not in tr
    assert float(tr["inclination"]["value"]) == pytest.approx(30.0)


def test_temperature_slope_is_a_site_when_fitted_and_a_constant_when_fixed(data, monkeypatch):
    monkeypatch.setattr(model_mod, "response_function", thin_disk_response)
    tr = _trace(_echofit(data, fit_temperature_slope=True))
    fn = tr["temperature_slope"]["fn"]
    assert (float(fn.low), float(fn.high)) == pytest.approx(model_mod.TEMPERATURE_SLOPE_PRIOR)
    tr = _trace(_echofit(data, fit_temperature_slope=True, fixed_params={"temperature_slope": 1.0}))
    assert "temperature_slope" not in _sampled(tr)
    assert float(tr["temperature_slope"]["value"]) == pytest.approx(1.0)


def test_gain_offset_and_background_priors_are_anchored_to_each_light_curve(data):
    ef = _echofit(data)
    tr = _trace(ef)
    scale = ef._model_kwargs()["sigma_drw_prior_scale"]
    for name, d in data["bands"].items():
        y = np.asarray(d["y"], dtype=np.float64)
        gain = tr[f"S_{name}"]["fn"]
        assert float(gain.loc) == pytest.approx(np.log(np.std(y) / scale), rel=1e-4)
        assert float(gain.scale) == pytest.approx(1.0)
        offset = tr[f"C_{name}"]["fn"]
        assert float(offset.loc) == pytest.approx(np.mean(y), rel=1e-4)
        assert float(offset.scale) == pytest.approx(10.0 * np.std(y), rel=1e-4)
        bg = tr[f"bg_{name}"]["fn"]
        while hasattr(bg, "base_dist"):  # unwrap .expand()/.to_event()
            bg = bg.base_dist
        assert float(np.max(bg.loc)) == 0.0
        assert float(np.mean(bg.scale)) == pytest.approx(model_mod.BACKGROUND_PRIOR_WIDTH * np.std(y), rel=1e-4)


def test_default_longest_driver_period_covers_twice_the_baseline_plus_tau_max(data):
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    t = np.concatenate([d["t"] for d in data["bands"].values()])
    span = t.max() - t.min()
    for tau_max in (None, 20.0):
        ef.build_grid(tau_max=tau_max)
        expected = PERIOD_MAX_FACTOR * (span + (0.5 * span if tau_max is None else tau_max))
        assert 2 * np.pi / float(np.min(ef.freqs)) == pytest.approx(expected, rel=1e-5)
    assert PERIOD_MAX_FACTOR == 2.0


@pytest.mark.parametrize("slope", [0.5, 0.75, 1.2])
def test_wien_radius_is_where_the_disc_has_the_wien_temperature(slope):
    log_mdot, wavelength, m_bh = 0.7, 5000.0, 2.0e8
    r = float(wien_radius(log_mdot, wavelength, m_bh, viscous_slope=slope))
    t1 = float(disk_t1_kelvin(log_mdot, m_bh))
    assert t1 * r ** -slope == pytest.approx(_WIEN_B_ANGSTROM_KELVIN / wavelength, rel=1e-4)
    if slope == 0.75:
        assert r == pytest.approx(float(lag_scaling(log_mdot, wavelength, m_bh)), rel=1e-4)
