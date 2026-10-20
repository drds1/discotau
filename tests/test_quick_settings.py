"""Quick checks of what each setting puts in the model, by tracing the model
once rather than fitting it (the fitting tier checks the same settings end to
end): which sample sites exist, and the priors anchored to the data."""

import numpy as np
import pytest
from numpyro import handlers

import discotau.model as model_mod
from discotau import EchoFit
from discotau.echofit import PERIOD_MAX_FACTOR
from discotau.forward_model import (
    disk_t1_kelvin, lag_scaling, log_mdot_from_t1, thin_disk_response, typical_log_mdot, wien_radius,
    _WIEN_B_ANGSTROM_KELVIN,
)
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
    log_mdot_msun_yr, wavelength, m_bh = 0.7, 5000.0, 2.0e8
    r = float(wien_radius(log_mdot_msun_yr, wavelength, m_bh, viscous_slope=slope))
    t1 = float(disk_t1_kelvin(log_mdot_msun_yr, m_bh))
    assert t1 * r ** -slope == pytest.approx(_WIEN_B_ANGSTROM_KELVIN / wavelength, rel=1e-4)
    if slope == 0.75:
        assert r == pytest.approx(float(lag_scaling(log_mdot_msun_yr, wavelength, m_bh)), rel=1e-4)


def _inclination_factor(ef, inclination_deg, marginalise_linear=False):
    """The type1 prior's factor at a given inclination (None if absent)."""
    kwargs = dict(ef._model_kwargs(), marginalise_linear=marginalise_linear)
    values = {"cos_inclination": np.cos(np.deg2rad(inclination_deg))}
    tr = handlers.trace(handlers.substitute(handlers.seed(reverberation_model, 0), values)).get_trace(**kwargs)
    site = tr.get("type1_inclination_prior")
    return None if site is None else float(site["fn"].log_factor)


def test_default_inclination_prior_is_unchanged(data):
    ef = _echofit(data)
    assert (ef.inclination_max, ef.inclination_prior) == (None, "isotropic")
    assert _inclination_factor(ef, 30.0) is None
    assert float(_trace(ef)["cos_inclination"]["fn"].low) == pytest.approx(np.cos(np.deg2rad(80.0)))


def test_inclination_max_sets_the_prior_bound(data):
    tr = _trace(_echofit(data, inclination_max=45.0))
    assert float(tr["cos_inclination"]["fn"].low) == pytest.approx(np.cos(np.deg2rad(45.0)))


@pytest.mark.parametrize("marginalise_linear", [False, True])
def test_type1_prior_tapers_the_inclination(data, marginalise_linear):
    ef = _echofit(data, inclination_prior="type1", type1_opening_deg=50.0, type1_width_deg=10.0)
    def expected(i):
        return -np.log1p(np.exp((i - 50.0) / 10.0))  # log of the logistic taper
    for i in (10.0, 50.0, 70.0):
        assert _inclination_factor(ef, i, marginalise_linear) == pytest.approx(expected(i), abs=1e-4)
    # Past the opening angle the prior falls off; well inside it, it is the isotropic one.
    assert _inclination_factor(ef, 70.0) < _inclination_factor(ef, 30.0) - 1.5
    assert _inclination_factor(ef, 10.0) == pytest.approx(0.0, abs=0.02)


def test_type1_prior_is_ignored_for_a_fixed_inclination(data):
    ef = _echofit(data, inclination_prior="type1", fixed_params={"inclination": 30.0})
    assert "type1_inclination_prior" not in _trace(ef)


@pytest.mark.parametrize("kwargs, message", [
    (dict(inclination_prior="type2"), "inclination_prior"),
    (dict(inclination_max=95.0), "inclination_max"),
    (dict(inclination_max=0.0), "inclination_max"),
    (dict(inclination_prior="type1", type1_width_deg=0.0), "type1"),
    (dict(inclination_prior="type1", type1_opening_deg=95.0), "type1"),
])
def test_bad_inclination_prior_settings_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        EchoFit(M_BH=1.0e8, **kwargs)


# -- the accretion rate in solar masses per year (October 2026) --

def test_t1_is_the_standard_thin_disc_temperature_at_one_light_day():
    # T_1**4 = 3 G M Mdot / (8 pi sigma R**3) at R = 1 light-day, in cgs, for
    # Fairall 9's mass and the 0.13 Msun/yr flux-matched rate (T_1 ~ 13.6 kK).
    G, c, sigma, msun, year = 6.674e-8, 2.99792458e10, 5.670374e-5, 1.98855e33, 3.15576e7
    m_bh, mdot = 2.55e8, 0.1316
    expected = (3 * G * m_bh * msun * mdot * msun / year / (8 * np.pi * sigma * (c * 86400.0) ** 3)) ** 0.25
    assert float(disk_t1_kelvin(np.log10(mdot), m_bh)) == pytest.approx(expected, rel=1e-4)
    assert expected == pytest.approx(13580.0, rel=2e-3)
    assert float(log_mdot_from_t1(expected, m_bh)) == pytest.approx(np.log10(mdot), abs=1e-4)


def test_lag_is_the_wien_radius_of_the_t1_disc():
    m_bh, log_mdot = 2.55e8, -0.88
    t1 = float(disk_t1_kelvin(log_mdot, m_bh))
    for wavelength in (1928.0, 5468.0):
        assert float(lag_scaling(log_mdot, wavelength, m_bh)) == pytest.approx(
            (t1 * wavelength / _WIEN_B_ANGSTROM_KELVIN) ** (4 / 3), rel=1e-4)


@pytest.mark.parametrize("m_bh", [1e6, 1e8, 1e10])
def test_typical_accretion_rate_is_about_half_a_per_cent_of_eddington(m_bh):
    # Eddington rate for radiative efficiency 0.1: L_Edd / (0.1 c**2), in Msun/yr.
    mdot_edd = 1.26e38 * m_bh / (0.1 * 2.99792458e10 ** 2) * 3.15576e7 / 1.98855e33
    ratio = 10 ** float(typical_log_mdot(m_bh)) / mdot_edd
    assert 0.004 < ratio < 0.006


def test_accretion_rate_prior_is_centred_on_the_typical_rate_and_reported_in_msun_per_year(data):
    tr = _trace(_echofit(data))
    site = tr["log_mdot_msun_yr"]
    assert float(site["fn"].loc) == pytest.approx(float(typical_log_mdot(1.0e8)))
    assert float(site["fn"].scale) == model_mod.LOG_MDOT_PRIOR_SD
    assert float(tr["mdot_msun_yr"]["value"]) == pytest.approx(10 ** float(site["value"]), rel=1e-5)


def test_the_old_dimensionless_name_is_refused_with_the_conversion(data):
    ef = _echofit(data, fixed_params={"log_mdot": 0.666})
    with pytest.raises(ValueError, match="typical_log_mdot"):
        ef._validate_before_fit()


def test_starts_are_centred_on_the_typical_rate(data):
    # NumPyro starts unconstrained sites uniformly in (-2, 2); the accretion rate's
    # start is shifted to that range about typical_log_mdot, as before the change.
    import jax
    from numpyro.infer.util import initialize_model

    import discotau.echofit as echofit_mod

    ef = _echofit(data)
    typical = float(typical_log_mdot(1.0e8))
    kwargs = ef._model_kwargs()
    for seed in range(3):
        nuts = initialize_model(jax.random.PRNGKey(seed), reverberation_model, model_kwargs=kwargs,
                                init_strategy=ef._fit_init_strategy(1, False)).param_info.z
        solve = echofit_mod._initialize_marginal_model(seed, kwargs, ef._init_strategy(1), ef.M_BH).param_info.z
        for z in (nuts, solve):
            assert -2.0 < float(z["log_mdot_msun_yr"]) - typical < 2.0
