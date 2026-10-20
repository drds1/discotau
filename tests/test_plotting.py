"""
Tests for the diagnostic plots added alongside the disk-response work:
plot_corner (and EchoFit's plot_corner_bands/plot_corner_free_lag
convenience wrappers around it) and plot_fourier_correlation, plus
reporting.generate_report's conditional inclusion of them (disk-parameter
corner only for lag_mode="physical" fits, free-lag corner only for
lag_mode="free" fits, band-offset corner and the Fourier correlation
heatmap always).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from discotau import plotting


def test_wavelength_to_colour_classifies_regimes_correctly():
    assert plotting.wavelength_to_colour(10.0) == "black"  # X-ray
    assert plotting.wavelength_to_colour(2000.0) == "darkviolet"  # UV
    assert plotting.wavelength_to_colour(20000.0) == "firebrick"  # IR+
    optical = plotting.wavelength_to_colour(5000.0)
    assert optical not in ("black", "darkviolet", "firebrick")
    assert isinstance(optical, tuple) and len(optical) == 3


def test_wavelength_to_colour_visible_range_is_blue_to_red():
    # Bluer (shorter) visible wavelengths should have a larger blue
    # component than red; redder (longer) ones the reverse.
    blue_ish = plotting.wavelength_to_colour(4200.0)
    red_ish = plotting.wavelength_to_colour(7000.0)
    assert blue_ish[2] > blue_ish[0]
    assert red_ish[0] > red_ish[2]


def test_band_colours_are_ordered_by_wavelength_and_use_wavelength_to_colour():
    bands = {
        "red_band": {"wavelength": 7000.0},
        "blue_band": {"wavelength": 4200.0},
    }
    ordered, colours = plotting._band_colours(bands)
    assert [name for name, _ in ordered] == ["blue_band", "red_band"]
    assert colours["blue_band"] == plotting.wavelength_to_colour(4200.0)
    assert colours["red_band"] == plotting.wavelength_to_colour(7000.0)


def test_plot_corner_shows_true_value_crosshair_and_raises_on_unknown_param():
    rng = np.random.default_rng(0)
    samples = {
        "log_mdot_msun_yr": rng.normal(0.0, 0.1, size=(2, 100)),
        "inclination": rng.normal(30.0, 3.0, size=(2, 100)),
    }
    fig, axes = plotting.plot_corner(samples, true_values={"log_mdot_msun_yr": 0.0, "inclination": 30.0})
    assert axes.shape == (2, 2)

    with pytest.raises(KeyError, match="not_a_param"):
        plotting.plot_corner(samples, param_names=("log_mdot_msun_yr", "not_a_param"))


def test_plot_corner_generalises_beyond_two_parameters():
    rng = np.random.default_rng(0)
    samples = {name: rng.normal(0.0, 1.0, size=(3, 50)) for name in ("a", "b", "c")}
    fig, axes = plotting.plot_corner(samples, param_names=("a", "b", "c"))
    assert axes.shape == (3, 3)


def test_plot_fourier_correlation_handles_pooled_or_by_chain_shapes():
    rng = np.random.default_rng(0)
    n_freq = 12
    freqs = np.geomspace(0.05, 3.0, n_freq)
    S_pooled = rng.normal(0.0, 1.0, size=(200, n_freq))
    C_by_chain = rng.normal(0.0, 1.0, size=(2, 100, n_freq))

    fig, axes = plotting.plot_fourier_correlation(S_pooled, C_by_chain, freqs)
    assert len(axes) == 2


def test_plot_fourier_correlation_handles_zero_variance_frequencies():
    """A frequency stuck at (or near) a constant value -- a too-short
    warmup, or a DRW prior that suppresses the highest frequencies enough
    for this to happen in practice -- makes a raw correlation coefficient
    a 0/0 division. This should come back as a plain 0.0, not numpy's nan
    (which would otherwise raise a RuntimeWarning and leave unexplained
    gaps in the heatmap)."""
    rng = np.random.default_rng(0)
    n_freq = 6
    freqs = np.geomspace(0.05, 3.0, n_freq)
    S = rng.normal(0.0, 1.0, size=(100, n_freq))
    S[:, 2] = 0.0  # exactly zero variance at one frequency

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning (e.g. the divide-by-zero) fails the test
        fig, axes = plotting.plot_fourier_correlation(S, S, freqs)

    # sanity check that this setup really does exercise the nan-producing
    # case (i.e. the test isn't accidentally vacuous)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_corr = np.corrcoef(S, rowvar=False)
    assert np.isnan(raw_corr[2, :]).any()


def test_plot_bof_one_line_per_chain():
    rng = np.random.default_rng(0)
    potential_energy = rng.normal(5.0, 1.0, size=(3, 50))
    fig, ax = plotting.plot_bof(potential_energy, checkpoint_every=20)
    # 3 chain lines plus 2 checkpoint-boundary axvlines (at x=20, x=40).
    assert len(ax.lines) == 5
    for c in range(3):
        np.testing.assert_allclose(ax.lines[c].get_ydata(), 2.0 * potential_energy[c])


def test_plot_bof_accepts_single_chain_1d_array():
    rng = np.random.default_rng(0)
    potential_energy = rng.normal(5.0, 1.0, size=50)
    fig, ax = plotting.plot_bof(potential_energy)
    assert len(ax.lines) == 1


def _tiny_physical_fit():
    from discotau.synthetic import generate_synthetic_dataset
    from discotau.echofit import EchoFit

    data = generate_synthetic_dataset(
        M_BH=1.0e8, bands={"g": 4770.0, "i": 7625.0}, n_obs_per_band=15,
        n_freq=6, n_tau=40, tau_max=30.0, noise_level=0.08, seed=0,
    )
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=6, n_tau=40)
    ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)
    return ef, data


@pytest.mark.fit
def test_echofit_plot_corner_bands_covers_every_band():
    ef, _ = _tiny_physical_fit()
    fig, axes = ef.plot_corner_bands()
    assert axes.shape == (4, 4)  # S_g, C_g, S_i, C_i


@pytest.mark.fit
def test_echofit_plot_corner_free_lag_raises_when_no_free_lag_bands():
    ef, _ = _tiny_physical_fit()
    with pytest.raises(ValueError, match="no lag_mode"):
        ef.plot_corner_free_lag()


@pytest.mark.fit
def test_echofit_plot_fourier_correlation_runs():
    ef, _ = _tiny_physical_fit()
    fig, axes = ef.plot_fourier_correlation()
    assert len(axes) == 2


@pytest.mark.fit
def test_echofit_plot_bof_runs_since_potential_energy_is_requested_by_default():
    ef, _ = _tiny_physical_fit()
    fig, ax = ef.plot_bof()
    assert len(ax.lines) == 1


@pytest.mark.fit
def test_echofit_plot_bof_raises_without_potential_energy():
    ef, _ = _tiny_physical_fit()
    ef._extra_fields_by_chain = {}
    with pytest.raises(RuntimeError, match="potential_energy"):
        ef.plot_bof()


@pytest.mark.fit
@pytest.mark.slow
def test_report_includes_disk_and_band_corners_but_not_free_lag_for_physical_fit(tmp_path):
    from discotau import reporting

    ef, data = _tiny_physical_fit()
    path = reporting.generate_report(ef, tmp_path, truth=data["truth"])
    html = path.read_text()

    assert "corner.png" in html
    assert "corner_bands.png" in html
    assert "fourier_correlation.png" in html
    assert "bof.png" in html
    assert "corner_free_lag.png" not in html
    for fname in ("corner.png", "corner_bands.png", "fourier_correlation.png", "bof.png"):
        assert (tmp_path / fname).exists()
    assert not (tmp_path / "corner_free_lag.png").exists()


@pytest.mark.fit
def test_report_includes_free_lag_corner_but_not_disk_corner_for_free_lag_only_fit(tmp_path):
    from discotau.synthetic import generate_free_lag_dataset
    from discotau.echofit import EchoFit
    from discotau import reporting

    data = generate_free_lag_dataset(
        lines={"line_a": 8.0}, n_obs_per_line=15, n_obs_driver=20,
        include_driver=True, n_freq=6, n_tau=40, tau_max=30.0, noise_level=0.08, seed=1,
    )
    ef = EchoFit(M_BH=None)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"], lag_mode="free")
    ef.add_driver_lightcurve(t=data["driver"]["t"], y=data["driver"]["y"], yerr=data["driver"]["yerr"])
    ef.build_grid(n_freq=6, n_tau=40, tau_max=30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ef.fit(rng_seed=0, num_warmup=10, num_samples=10, progress_bar=False)

    path = reporting.generate_report(ef, tmp_path, truth=data["truth"])
    html = path.read_text()

    assert "corner.png" not in html
    assert "corner_free_lag.png" in html
    assert "corner_bands.png" in html
    assert "fourier_correlation.png" in html


def _two_basin_landscape():
    """A nested_laplace_result-like dict with two basins (no fit needed)."""
    def logp(lm, cosi):
        a = -0.5 * (((lm - 2.0) / 0.3) ** 2 + ((cosi - 0.9) / 0.05) ** 2)
        b = -0.5 * (((lm - 4.5) / 0.4) ** 2 + ((cosi - 0.25) / 0.08) ** 2) - 3.0
        return np.logaddexp(a, b)

    scout = (np.linspace(-10.0, 15.0, 26), np.linspace(0.17, 1.0, 12))
    fine = (np.linspace(1.0, 5.5, 20), np.linspace(0.17, 1.0, 15))
    grid = lambda axes: logp(*np.meshgrid(*axes, indexing="ij"))  # noqa: E731
    return dict(names=["log_mdot_msun_yr", "cos_inclination"], passes=[dict(axes=scout, logp=grid(scout))],
                fine_axes=fine, fine_logp=grid(fine))


def test_plot_landscape_3d_draws_both_grids_as_surfaces():
    from discotau.plotting import plot_landscape_3d

    result = _two_basin_landscape()
    for height in ("log", "density"):
        fig, axes = plot_landscape_3d(result, height=height, truth={"log_mdot_msun_yr": 2.0, "inclination": 25.8})
        assert len(axes) == 2 and all(ax.name == "3d" for ax in axes)
        assert all(ax.collections for ax in axes)  # the surfaces and floor contours
        plt.close(fig)
    with pytest.raises(ValueError, match="height"):
        plot_landscape_3d(result, height="silly")
    # Any gridded pair works, labelled by its own names.
    fig, axes = plot_landscape_3d(dict(result, names=["log_mdot_msun_yr", "temperature_slope"]))
    assert axes[1].get_xlabel() == "temperature_slope" and axes[1].get_ylabel() == "log_mdot_msun_yr"
    plt.close(fig)
    with pytest.raises(ValueError, match="two gridded"):
        plot_landscape_3d(dict(result, names=["log_mdot_msun_yr"]))


def test_plot_sample_landscape_3d_works_for_any_pair():
    from discotau.plotting import plot_sample_landscape_3d

    rng = np.random.default_rng(0)
    samples = {"log_mdot_msun_yr": rng.normal(2.0, 0.1, 4000), "inclination": rng.uniform(10, 60, 4000),
               "dce_fraction_V": rng.beta(6, 4, 4000), "M_fixed": np.full(4000, 3.0)}
    fig, ax = plot_sample_landscape_3d(samples, "cos_inclination", "log_mdot_msun_yr", truth={"log_mdot_msun_yr": 2.0, "inclination": 30})
    assert ax.name == "3d" and ax.get_xlabel() == "cos(inclination)"
    plt.close(fig)
    fig, ax = plot_sample_landscape_3d(samples, "dce_fraction_V", "log_mdot_msun_yr", height="density")
    assert ax.get_xlabel() == "dce_fraction_V"
    plt.close(fig)
    with pytest.raises(ValueError, match="fixed"):
        plot_sample_landscape_3d(samples, "M_fixed", "log_mdot_msun_yr")
    with pytest.raises(KeyError):
        plot_sample_landscape_3d(samples, "nope", "log_mdot_msun_yr")


def test_lightcurve_fits_split_a_second_reprocessors_response():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 50, 40)
    tau = np.linspace(0, 20, 60)
    bands = {name: dict(t=t, y=rng.normal(size=40), yerr=np.full(40, 0.1), wavelength=wl, lag_mode="physical")
             for name, wl in (("g", 4770.0), ("i", 7625.0))}
    y_pred = {name: rng.normal(size=(30, 80)) for name in bands}
    disc = np.exp(-tau / 2.0)[None, :].repeat(30, 0) * 0.3
    second = np.exp(-0.5 * ((tau - 10.0) / 3.0) ** 2)[None, :].repeat(30, 0) * 0.05
    psi = {"g": disc + second, "i": disc}
    fig, axes = plotting.plot_lightcurve_fits(bands, np.linspace(0, 50, 80), y_pred, tau, psi,
                                              psi_component_samples={"g": dict(disc=disc, second=second)})
    labels = [line.get_label() for line in axes[0, 1].get_lines()]
    assert "disc" in labels and axes[0, 1].get_legend() is not None
    assert "disc" not in [line.get_label() for line in axes[1, 1].get_lines()]  # no second reprocessor in i
    plt.close(fig)
