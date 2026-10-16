"""
Tests for EchoFit(title=...)'s on-disk run outputs: checkpointing during a
fit, and resuming a fit that was interrupted mid-run. These check the
mechanics (right files, right sample counts, no duplicated/dropped samples
across the resume boundary) rather than statistical recovery -- settings
here are small/fast on purpose, unlike tests/test_recovery.py.
"""

import numpy as np
import pytest

from discotau.synthetic import generate_synthetic_dataset
from discotau.echofit import EchoFit
import discotau.echofit as echofit_mod


def _make_synthetic(seed=0):
    return generate_synthetic_dataset(
        M_BH=1.0e9,
        bands={"g": 4770.0, "i": 7625.0},
        t_span=200.0, n_obs_per_band=20, n_freq=8, n_tau=40, tau_max=50.0,
        noise_level=0.08, seed=seed,
    )


def _build_echofit(data, **kwargs):
    ef = EchoFit(M_BH=data["truth"]["M_BH"], **kwargs)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=8, n_tau=40)
    return ef


def test_title_run_writes_expected_outputs(tmp_path):
    data = _make_synthetic()
    ef = _build_echofit(data, title="unit_test_agn", output_dir=str(tmp_path))
    ef.fit(rng_seed=0, num_warmup=20, num_samples=20, checkpoint_every=20, progress_bar=False)

    assert ef.run_dir is not None
    assert ef.run_dir.parent.name == "unit_test_agn"
    assert ef.run_dir.name.startswith("run_")

    for fname in (
        "manifest.json", "data.npz", "grid.npz", "chains.npz", "report.html",
        "raw_lightcurves.png", "lightcurve_fits.png", "power_spectrum.png", "mcmc_diagnostics.png",
        "corner.png", "corner_bands.png", "fourier_correlation.png", "bof.png",
        "checkpoint/samples.npz", "checkpoint/extra_fields.npz", "checkpoint/state.pkl",
    ):
        assert (ef.run_dir / fname).exists(), f"missing {fname}"

    assert len(ef.samples["log_mdot"]) == 20


@pytest.mark.slow
def test_resume_continues_an_interrupted_fit(tmp_path, monkeypatch):
    data = _make_synthetic()
    ef = _build_echofit(data, title="resume_unit_test", output_dir=str(tmp_path))

    # Simulate the process being killed right after the first checkpoint by
    # raising out of the on_chunk_done callback once 20 samples are saved.
    real_chunked = echofit_mod.run_mcmc_chunked
    raised = {"done": False}

    def flaky_chunked(*args, **kwargs):
        orig_cb = kwargs["on_chunk_done"]

        def cb(mcmc, last_state, n_done_total):
            orig_cb(mcmc, last_state, n_done_total)
            if n_done_total >= 20 and not raised["done"]:
                raised["done"] = True
                raise KeyboardInterrupt("simulated kill mid-run")

        kwargs["on_chunk_done"] = cb
        return real_chunked(*args, **kwargs)

    monkeypatch.setattr(echofit_mod, "run_mcmc_chunked", flaky_chunked)

    with pytest.raises(KeyboardInterrupt):
        ef.fit(rng_seed=0, num_warmup=15, num_samples=50, checkpoint_every=20, progress_bar=False)

    monkeypatch.setattr(echofit_mod, "run_mcmc_chunked", real_chunked)

    checkpoint_samples = np.load(ef.run_dir / "checkpoint" / "samples.npz")
    assert len(checkpoint_samples["log_mdot"]) == 20

    # Resume in a *fresh* instance, like a new process would.
    ef2 = EchoFit.resume("resume_unit_test", output_dir=str(tmp_path))
    assert ef2.run_dir == ef.run_dir  # same run, not a new timestamped one
    ef2.fit(progress_bar=False)  # no args -> reuses the original run's config

    assert len(ef2.samples["log_mdot"]) == 50
    final_checkpoint = np.load(ef.run_dir / "checkpoint" / "samples.npz")
    assert len(final_checkpoint["log_mdot"]) == 50

    chains = np.load(ef.run_dir / "chains.npz")
    assert len(chains["log_mdot"]) == 50

    # The pre-kill samples should be preserved verbatim (loaded from the
    # checkpoint, not re-simulated with different draws), and the resumed
    # run should have actually continued sampling rather than, say,
    # padding with repeats of the same 20 samples.
    np.testing.assert_array_equal(
        checkpoint_samples["log_mdot"], final_checkpoint["log_mdot"][:20]
    )
    assert not np.array_equal(
        final_checkpoint["log_mdot"][:20], final_checkpoint["log_mdot"][20:40]
    )


@pytest.mark.slow
def test_report_every_refreshes_report_mid_fit(tmp_path, monkeypatch):
    """report_every should write report.html (and its PNGs) at least once
    before the fit finishes, not only at the very end -- checked by
    watching for report.html to already exist from inside on_chunk_done
    partway through the run, before the final chunk completes."""
    data = _make_synthetic()
    ef = _build_echofit(data, title="report_every_unit_test", output_dir=str(tmp_path))

    seen_mid_fit = {"exists": False}
    real_chunked = echofit_mod.run_mcmc_chunked

    def watching_chunked(*args, **kwargs):
        orig_cb = kwargs["on_chunk_done"]

        def cb(mcmc, last_state, n_done_total):
            orig_cb(mcmc, last_state, n_done_total)
            if n_done_total < 40:
                seen_mid_fit["exists"] = seen_mid_fit["exists"] or (ef.run_dir / "report.html").exists()

        kwargs["on_chunk_done"] = cb
        return real_chunked(*args, **kwargs)

    monkeypatch.setattr(echofit_mod, "run_mcmc_chunked", watching_chunked)
    ef.fit(
        rng_seed=0, num_warmup=10, num_samples=40, checkpoint_every=20,
        report_every=20, progress_bar=False,
    )

    assert seen_mid_fit["exists"]
    assert (ef.run_dir / "report.html").exists()
    assert len(ef.samples["log_mdot"]) == 40


@pytest.mark.slow
def test_resume_of_already_complete_run_is_a_no_op(tmp_path):
    """Regression test: resuming a run that already reached its target
    sample count used to crash (run_mcmc_chunked indexed into an empty
    chunk list) instead of being a harmless no-op."""
    data = _make_synthetic()
    ef = _build_echofit(data, title="complete_run", output_dir=str(tmp_path))
    ef.fit(rng_seed=0, num_warmup=15, num_samples=20, checkpoint_every=20, progress_bar=False)
    assert len(ef.samples["log_mdot"]) == 20

    ef2 = EchoFit.resume("complete_run", output_dir=str(tmp_path))
    ef2.fit(progress_bar=False)
    assert len(ef2.samples["log_mdot"]) == 20
    np.testing.assert_array_equal(ef.samples["log_mdot"], ef2.samples["log_mdot"])
