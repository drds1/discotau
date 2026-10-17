"""Every setting survives a checkpointed run's resume, without fitting.

A setting lost on ``EchoFit.resume()`` silently changes the model partway
through a run, and it happened five times as features were added
(fixed_params, the error model, the DRW prior, the temperature slope, the SED
settings; CLAUDE.md decisions #15, #18, #19, #22, #24). The checkpointed
``fit()`` writes every run file before it samples, so this test stops it at the
sampler, resumes, and compares. Add each new setting here."""

import numpy as np
import pytest

import discotau.echofit
import discotau.model as model_mod
from discotau import EchoFit
from discotau.forward_model import thin_disk_response
from discotau.synthetic import generate_synthetic_dataset


class _StopBeforeSampling(Exception):
    pass


def _stop(*args, **kwargs):
    raise _StopBeforeSampling


SETTINGS = dict(
    fixed_params={"sigma_jitter_g": 0.0},
    drw_prior=True,
    marginalise_linear=True,
    fit_temperature_slope=True,
    redshift=0.047,
    flux_unit="f_lambda",
    flux_scale=1e-15,
    ebv_galactic=0.03,
    sed_analysis=True,
    sed_options={"host_band": "g", "include_irradiation": False},
    inclination_max=60.0,
    inclination_prior="type1",
    type1_opening_deg=40.0,
    type1_width_deg=7.0,
)
FIT = dict(num_warmup=7, num_samples=11, checkpoint_every=5, max_tree_depth=4, dense_mass=False, rng_seed=3)
BANDS = {  # name: (lag_mode, fit_error_model, diffuse_continuum, background_order)
    "g": ("physical", True, True, 2),
    "i": ("physical", False, False, 0),
    "Hb": ("free", True, False, 1),
}
DRIVER = dict(fit_error_model=True, background_order=3)


def test_every_setting_survives_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(model_mod, "response_function", thin_disk_response)  # fit_temperature_slope needs it
    monkeypatch.setattr(discotau.echofit, "run_mcmc_chunked", _stop)
    data = generate_synthetic_dataset(bands={"g": 4770.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=25,
                                      n_freq=8, n_tau=40, seed=0)
    ef = EchoFit(M_BH=1.0e8, title="resume_settings", output_dir=str(tmp_path), **SETTINGS)
    for name, (lag_mode, error_model, dce, bg) in BANDS.items():
        d = data["bands"]["g" if name == "Hb" else name]
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode=lag_mode,
                          fit_error_model=error_model, diffuse_continuum=dce, background_order=bg)
    d = data["bands"]["i"]
    ef.add_driver_lightcurve(t=d["t"], y=d["y"], yerr=d["yerr"], **DRIVER)
    ef.build_grid(n_freq=8, n_tau=40)
    with pytest.raises(_StopBeforeSampling):
        ef.fit(progress_bar=False, **FIT)

    back = EchoFit.resume("resume_settings", output_dir=str(tmp_path))
    for key, value in SETTINGS.items():
        assert getattr(back, key) == value, key
    for key in ("num_warmup", "num_samples", "checkpoint_every", "max_tree_depth", "dense_mass", "rng_seed"):
        assert back._fit_config[key] == FIT[key], key
    for name, (lag_mode, error_model, dce, bg) in BANDS.items():
        b = back.bands[name]
        assert (b["lag_mode"], b["fit_error_model"], b["diffuse_continuum"], b["background_order"]) == \
            (lag_mode, error_model, dce, bg), name
        np.testing.assert_array_equal(b["y"], ef.bands[name]["y"])
    assert back.driver_data is not None
    for key, value in DRIVER.items():
        assert back.driver_data[key] == value, key
    np.testing.assert_allclose(back.freqs, ef.freqs)
    np.testing.assert_allclose(back.tau_grid, ef.tau_grid)
    # The resumed model has the same sample sites as the original.
    assert set(back._valid_fixed_param_names()) == set(ef._valid_fixed_param_names())
