"""
Tests for the per-light-curve error model (CLAUDE.md decision #18):
``add_lightcurve(..., fit_error_model=True)`` / ``add_driver_lightcurve(...,
fit_error_model=True)`` add ``sigma_scale_{name}``/``sigma_jitter_{name}``
nuisance parameters that rescale that light curve's reported ``yerr``
(``sigma_eff = sqrt((sigma_scale*yerr)**2 + sigma_jitter**2)``), a direct
adaptation of the author's PhD-era CREAM Fortran code's own
``sigexpand``/``varexpand`` parameters. Off by default, per light curve.
"""

import numpy as np
import pytest

from discotau.synthetic import generate_synthetic_dataset, generate_free_lag_dataset
from discotau.echofit import EchoFit


def _dataset():
    return generate_synthetic_dataset(
        M_BH=1.0e8, bands={"g": 4770.0, "i": 7625.0}, n_obs_per_band=25,
        n_freq=8, n_tau=40, tau_max=30.0, noise_level=0.05, seed=0,
    )


def test_error_model_off_by_default_no_new_sites():
    data = _dataset()
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)

    assert "sigma_scale_g" not in ef.samples
    assert "sigma_jitter_g" not in ef.samples


def test_error_model_on_for_one_band_adds_only_that_bands_sites():
    data = _dataset()
    ef = EchoFit(M_BH=1.0e8)
    for i, (name, d) in enumerate(data["bands"].items()):
        # only the first band opts in -- confirms this is a genuinely
        # per-light-curve switch, not a global one.
        ef.add_lightcurve(
            name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"],
            fit_error_model=(i == 0),
        )
    ef.build_grid(n_freq=8, n_tau=40)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)

    on_band, off_band = list(data["bands"])[0], list(data["bands"])[1]
    assert f"sigma_scale_{on_band}" in ef.samples
    assert f"sigma_jitter_{on_band}" in ef.samples
    assert f"sigma_scale_{off_band}" not in ef.samples
    assert f"sigma_jitter_{off_band}" not in ef.samples
    assert np.all(np.asarray(ef.samples[f"sigma_scale_{on_band}"]) > 0)
    assert np.all(np.asarray(ef.samples[f"sigma_jitter_{on_band}"]) >= 0)


def test_error_model_can_be_fixed_via_fixed_params():
    data = _dataset()
    ef = EchoFit(
        M_BH=1.0e8,
        fixed_params={"sigma_scale_g": 1.0, "sigma_jitter_g": 0.0},
    )
    for name, d in data["bands"].items():
        ef.add_lightcurve(
            name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"],
            fit_error_model=(name == "g"),
        )
    ef.build_grid(n_freq=8, n_tau=40)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)

    assert np.all(ef.samples["sigma_scale_g"] == 1.0)
    assert np.all(ef.samples["sigma_jitter_g"] == 0.0)


def test_error_model_unknown_fixed_key_still_validated():
    """fixed_params validation (_valid_fixed_param_names) must know about
    sigma_scale_{name}/sigma_jitter_{name} only when that light curve's
    error model is actually on, not unconditionally."""
    data = _dataset()
    ef = EchoFit(M_BH=1.0e8, fixed_params={"sigma_scale_g": 1.0})
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    with pytest.raises(ValueError, match="sigma_scale_g"):
        ef.fit(num_warmup=5, num_samples=5, progress_bar=False)


def test_error_model_on_driver_lightcurve():
    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_obs_driver=20,
        include_driver=True, n_freq=6, n_tau=30, tau_max=25.0, noise_level=0.05, seed=0,
    )
    ef = EchoFit(M_BH=None)
    d = data["bands"]["line_a"]
    ef.add_lightcurve("line_a", wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(
        t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"], fit_error_model=True,
    )
    ef.build_grid(n_freq=6, n_tau=30, tau_max=25.0)
    ef.fit(num_warmup=5, num_samples=5, progress_bar=False)

    assert "sigma_scale_driver" in ef.samples
    assert "sigma_jitter_driver" in ef.samples
    assert "sigma_scale_line_a" not in ef.samples  # only the driver opted in


def test_error_model_persists_across_resume(tmp_path):
    """fit_error_model must survive the checkpointed save/resume round trip
    (run_manager.save_bands_npz/save_driver_npz) -- found (and fixed) as
    the same class of bug decision #15 hit with fixed_params: both were
    previously hardcoded to only a fixed set of keys and would silently
    drop anything new."""
    data = _dataset()
    ef = EchoFit(M_BH=1.0e8, title="error_model_resume_test", output_dir=str(tmp_path))
    for name, d in data["bands"].items():
        ef.add_lightcurve(
            name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"],
            fit_error_model=(name == "g"),
        )
    ef.build_grid(n_freq=8, n_tau=40)
    ef.fit(rng_seed=0, num_warmup=5, num_samples=10, checkpoint_every=5, progress_bar=False)
    assert "sigma_scale_g" in ef.samples

    ef2 = EchoFit.resume("error_model_resume_test", output_dir=str(tmp_path))
    assert ef2.bands["g"]["fit_error_model"] is True
    assert ef2.bands["i"]["fit_error_model"] is False
    ef2.fit(progress_bar=False)
    assert "sigma_scale_g" in ef2.samples
    assert "sigma_scale_i" not in ef2.samples
