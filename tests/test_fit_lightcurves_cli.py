"""scripts/fit_lightcurves.py, the command-line tool: each flag reaches the
EchoFit it sets up, and bad input is refused, including two bands reading one
file (on macOS's case-insensitive filesystem "r_band.txt" and "R_band.txt" are
the same file; NGC 5548's "13-band" fits once read R and I twice). The tool is
stopped where it would fit, so nothing is fitted."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from discotau.echofit import EchoFit

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_lightcurves.py"


class _Captured(Exception):
    def __init__(self, ef, kwargs):
        super().__init__()
        self.ef, self.kwargs = ef, kwargs


@pytest.fixture
def cli(monkeypatch):
    spec = importlib.util.spec_from_file_location("fit_lightcurves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def capture(self, **kwargs):
        raise _Captured(self, kwargs)

    monkeypatch.setattr(EchoFit, "fit", capture)
    monkeypatch.setattr(EchoFit, "optimise", capture)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["fit_lightcurves.py", *map(str, args)])
        with pytest.raises(_Captured) as info:
            module.main()
        return info.value.ef, info.value.kwargs

    run.module = module
    return run


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(0)
    paths = {}
    for name in ("g", "i", "hb", "x"):
        t = np.sort(rng.uniform(0, 60, 40))
        path = tmp_path / f"{name}.txt"
        np.savetxt(path, np.column_stack([t, 1 + 0.1 * rng.normal(size=40), np.full(40, 0.02)]))
        paths[name] = path
    return paths


def test_flags_reach_the_echofit(cli, files):
    ef, kwargs = cli("--m-bh", 1e8, "--band", "g", 4770, files["g"], "--band", "i", 7625, files["i"],
                     "--fit-error-model", "--diffuse-continuum", "--background-order", 2,
                     "--diagonal-mass", "--max-tree-depth", 6, "--n-tau", 120, "--tau-max", 30)
    for name in ("g", "i"):
        b = ef.bands[name]
        assert (b["fit_error_model"], b["diffuse_continuum"], b["background_order"], b["lag_mode"]) == \
            (True, True, 2, "physical")
    assert len(ef.tau_grid) == 120 and float(ef.tau_grid[-1]) == pytest.approx(30.0)
    assert kwargs["dense_mass"] is False and kwargs["max_tree_depth"] == 6


def test_diffuse_continuum_can_name_bands_and_defaults_off(cli, files):
    ef, _ = cli("--m-bh", 1e8, "--band", "g", 4770, files["g"], "--band", "i", 7625, files["i"],
                "--diffuse-continuum", "g")
    assert ef.bands["g"]["diffuse_continuum"] and not ef.bands["i"]["diffuse_continuum"]
    ef, kwargs = cli("--m-bh", 1e8, "--band", "g", 4770, files["g"])
    assert not ef.bands["g"]["diffuse_continuum"]
    assert ef.bands["g"]["background_order"] == 1  # the default linear background
    assert kwargs["dense_mass"] is True


def test_free_lag_bands_and_driver(cli, files):
    ef, _ = cli("--free-lag-band", "Hb", 4861, files["hb"], "--driver", files["x"])
    assert ef.bands["Hb"]["lag_mode"] == "free"
    assert ef.driver_data is not None


def test_optimise_picks_the_solver(cli, files):
    _, kwargs = cli("--m-bh", 1e8, "--band", "g", 4770, files["g"], "--optimise")
    assert kwargs["method"] == "nested_laplace"
    _, kwargs = cli("--m-bh", 1e8, "--band", "g", 4770, files["g"], "--optimise", "--laplace")
    assert kwargs["method"] == "laplace"


def test_two_bands_reading_one_file_are_refused(cli, files, tmp_path):
    with pytest.raises(SystemExit, match="same file"):
        cli.module._check_distinct_files([("r", 6200, str(files["g"])), ("R", 6400, str(files["g"]))])
    other_name = tmp_path / "g_second_name.txt"
    os.link(files["g"], other_name)  # a second name for the same file, as on a case-insensitive disk
    with pytest.raises(SystemExit, match="same file"):
        cli.module._check_distinct_files([("g", 4770, str(files["g"])), ("G", 4770, str(other_name))])
    cli.module._check_distinct_files([("g", 4770, str(files["g"])), ("i", 7625, str(files["i"]))])


@pytest.mark.parametrize("args, message", [
    (["--resume"], "--resume requires --title"),
    (["--m-bh", "1e8"], "Add at least one"),
])
def test_bad_arguments_are_refused(cli, monkeypatch, args, message):
    monkeypatch.setattr(sys, "argv", ["fit_lightcurves.py", *args])
    with pytest.raises(SystemExit, match=message):
        cli.module.main()


def test_unknown_diffuse_continuum_band_is_refused(cli, files, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_lightcurves.py", "--m-bh", "1e8", "--band", "g", "4770", str(files["g"]),
                                      "--diffuse-continuum", "z"])
    with pytest.raises(SystemExit, match="unknown band"):
        cli.module.main()
