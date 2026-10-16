"""
Tests for three related EchoFit/model.py features, all from the same
round of feedback:

* ``inclination`` is sampled uniform in ``cos(inclination)`` (the
  isotropic-orientation prior), not uniform in ``inclination`` itself --
  see model.py's "inclination note".
* ``EchoFit(fixed_params={...})`` holds any scalar site fixed instead of
  inferring it, generalising decision #3 ("M_BH is always fixed") to any
  parameter -- see model.py's "fixed-parameter note".
* ``EchoFit._init_strategy`` gives NUTS data-anchored starting guesses for
  each band's S_band/C_band, mirroring the author's PhD-era CREAM Fortran
  code's own initialisation.
"""

import warnings

import numpy as np
import pytest

from discotau.synthetic import generate_synthetic_dataset, generate_free_lag_dataset
from discotau.echofit import EchoFit


def _physical_dataset():
    return generate_synthetic_dataset(
        M_BH=1.0e8, bands={"g": 4770.0, "i": 7625.0}, n_obs_per_band=25,
        n_freq=8, n_tau=40, tau_max=30.0, noise_level=0.05, seed=0,
    )


def _build_physical_echofit(data, **kwargs):
    ef = EchoFit(M_BH=1.0e8, **kwargs)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    return ef


@pytest.mark.slow
def test_inclination_is_sampled_via_uniform_cos_inclination():
    ef = _build_physical_echofit(_physical_dataset())
    ef.fit(rng_seed=0, num_warmup=15, num_samples=15, progress_bar=False)

    assert "cos_inclination" in ef.samples
    cos_incl, incl = ef.samples["cos_inclination"], ef.samples["inclination"]
    np.testing.assert_allclose(np.degrees(np.arccos(cos_incl)), incl, atol=1e-3)
    # cos_inclination's own prior support is [cos(80deg), 1] -- inclination
    # (its arccos, in degrees) must stay within [0, 80] as a result.
    assert np.all((incl >= 0.0) & (incl <= 80.0))


def test_fixed_inclination_skips_cos_inclination_entirely():
    ef = _build_physical_echofit(_physical_dataset(), fixed_params={"inclination": 0.0})
    ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)

    assert "cos_inclination" not in ef.samples
    assert np.all(ef.samples["inclination"] == 0.0)
    # log_mdot is not fixed, so it should still show real posterior spread.
    assert ef.samples["log_mdot"].std() > 0.0


def test_fixed_free_lag_tau_and_offset():
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=20, n_obs_driver=25,
        include_driver=True, n_freq=8, n_tau=40, tau_max=30.0, noise_level=0.05, seed=1,
    )
    ef = EchoFit(M_BH=None, fixed_params={"tau_line_a": 8.0, "C_line_a": 0.0})
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    ef.build_grid(n_freq=8, n_tau=40, tau_max=30.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)

    assert np.all(ef.samples["tau_line_a"] == 8.0)
    assert np.all(ef.samples["C_line_a"] == 0.0)
    assert ef.samples["S_line_a"].std() > 0.0  # not fixed, still inferred


def test_unknown_fixed_param_key_raises():
    ef = _build_physical_echofit(_physical_dataset(), fixed_params={"not_a_real_site": 1.0})
    with pytest.raises(ValueError, match="not_a_real_site"):
        ef.fit(rng_seed=0, num_warmup=5, num_samples=5, progress_bar=False)


def test_valid_fixed_param_names_covers_physical_and_driver_sites():
    ef = _build_physical_echofit(_physical_dataset(), drw_prior=True)
    valid = ef._valid_fixed_param_names()
    assert {"sigma_drw", "tau_drw", "log_mdot", "inclination", "S_g", "C_g", "S_i", "C_i"} <= valid
    assert "tau_g" not in valid  # physical-mode bands have no tau_{band} site


def test_tau_drw_only_valid_fixed_param_when_drw_prior_is_on():
    """tau_drw isn't even a real site under the default drw_prior=False
    (random-walk) prior -- see CLAUDE.md decision #19 -- so it shouldn't
    be accepted as a fixed_params key either."""
    ef_rw = _build_physical_echofit(_physical_dataset())  # drw_prior defaults False
    assert "tau_drw" not in ef_rw._valid_fixed_param_names()
    assert "sigma_drw" in ef_rw._valid_fixed_param_names()


def test_init_strategy_matches_data_mean_and_std_ratio():
    data = _physical_dataset()
    ef = _build_physical_echofit(data)
    scale = ef._sigma_drw_prior_scale()
    values = ef._init_strategy().keywords["values"]

    for name, d in ef.bands.items():
        assert values[f"C_{name}"] == pytest.approx(np.mean(d["y"]))
        assert values[f"S_{name}"] == pytest.approx(np.std(d["y"]) / scale)


def test_init_strategy_excludes_fixed_sites():
    ef = _build_physical_echofit(_physical_dataset(), fixed_params={"C_g": 0.0})
    values = ef._init_strategy().keywords["values"]
    assert "C_g" not in values
    assert "S_g" in values  # not fixed, still gets a starting guess


def test_init_strategy_disabled_for_multi_chain_fits():
    """Regression test: init_to_value gives every chain the exact same
    starting point, which defeats Gelman-Rubin R-hat convergence checking
    for multi-chain fits (confirmed directly: this made
    test_free_lag_recovery_with_driver_anchor's 4-chain R-hat blow up to
    ~1000 on the free-lag tau_{band} sites, down from ~1.0 with this
    disabled) -- must stay off whenever num_chains != 1."""
    ef = _build_physical_echofit(_physical_dataset())
    assert ef._init_strategy(num_chains=1) is not None
    assert ef._init_strategy(num_chains=4) is None


@pytest.mark.slow
def test_resume_persists_and_restores_fixed_params(tmp_path):
    data = _physical_dataset()
    ef = EchoFit(M_BH=1.0e8, title="resume_fixed_test", output_dir=str(tmp_path), fixed_params={"inclination": 0.0})
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    ef.fit(rng_seed=0, num_warmup=10, num_samples=20, checkpoint_every=10, progress_bar=False)
    assert np.all(ef.samples["inclination"] == 0.0)

    ef2 = EchoFit.resume("resume_fixed_test", output_dir=str(tmp_path))
    assert ef2.fixed_params == {"inclination": 0.0}
    ef2.fit(progress_bar=False)
    assert np.all(ef2.samples["inclination"] == 0.0)
    assert "cos_inclination" not in ef2.samples
