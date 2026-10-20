"""
make_fit_animation.py
======================

Renders a GIF of the fit "taking shape" sample by sample, early in a NUTS
chain, as the driver's Fourier coefficients and disk parameters are still
finding their way -- purely a demo/visualisation aid (see the animation
embedded in README.md), not something a real fit needs. Reuses the same
closed-form echo evaluation as ``EchoFit.plot_lightcurve_fits``
(``forward_model.driver_at``/``transfer_coeffs``/``compute_echo``), just
evaluated per-sample in chain order instead of as a posterior-predictive
summary over a random subset.

Three columns, one row per band plus a shared top row (the light-curve
column is wider than the other two -- ``width_ratios``):

* top-left: the inferred driving light curve X(t).
* top-middle: a face-on, illustrative view of the accretion disk, each
  ring drawn in the true colour of a blackbody at its
  ``forward_model.disk_temperature_profile`` (Shakura-Sunyaev viscous)
  temperature, via ``plotting.blackbody_to_colour`` -- blue-white at the
  hot centre, through white, to orange-red in the cool outer disk, on a
  black background. The colour-to-temperature mapping (a log-temperature
  colour bar in Kelvin) is fixed across frames, so a change in colour is a
  genuine change in temperature as log_mdot_msun_yr varies, not a rescaled colour
  map. Geometry is fixed; only the colours change per frame. The title reports the
  frame's current log_mdot_msun_yr/inclination values.
* top-right: a schematic side-on view showing the disk plane as a line
  through the centre, tilting by the inclination angle (vertical at
  inclination=0/face-on, rotating toward horizontal/aligned with the
  line of sight as inclination approaches edge-on). The observer (a
  small eye-on-a-sphere icon) and its dashed line of sight are fixed --
  only the disk line rotates, so the tilt is unambiguous rather than
  conflated with the observer's own position moving.
* per band: its echo light curve (left) and response function psi(tau)
  (middle, x-axis capped at a shared limit computed by
  ``plotting.response_function_xlim`` from wherever the widest band's
  response has actually decayed to negligible area -- the response itself
  is always much narrower than the full lag grid it's evaluated on, and
  this is the same mechanism ``plot_lightcurve_fits`` uses, not an
  independently-chosen/hardcoded value).
* bottom: Badness-of-Fit (``2 * potential_energy``, matching
  ``plotting.plot_bof``) vs. sample number, drawn as a growing trace that
  extends one point per frame -- should trend down then flatten as the
  chain converges.

Every light-curve/psi panel (driver included) also shows 68%/95% credible
envelopes, matching the shaded bands on the standard (non-animated)
``plot_lightcurve_fits``, but computed cumulatively per frame -- frame i's
envelope only uses samples up to i, not the full run -- so it starts wide
(next to no constraint from 1-2 samples) and narrows towards the converged
posterior's own width as the chain accumulates more of them; see
``_expanding_percentiles``. All light-curve/psi/BOF panels have gridlines
(``alpha=0.6`` -- the matplotlib default ``alpha=0.25`` on its default grey
turned out to be nearly invisible once actually rendered, not just subtle).

Both disk panels are a genuinely simplified 2-D schematic, not a 3-D/
raytraced render, and always use the viscous thin-disk profile for the
picture regardless of which response function the fit itself used, since
only the thin-disk family has a literal disk geometry to show.

A short warmup on purpose: the point is to see the chain still finding the
answer, not a converged posterior from frame one.

Usage
-----
    python scripts/make_fit_animation.py
    python scripts/make_fit_animation.py --num-frames 200 --fps 20 --outdir docs/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

from discotau import model as _model
from discotau.echofit import EchoFit
from discotau.forward_model import (
    transfer_coeffs, compute_echo, driver_at,
    disk_temperature_profile, _schwarzschild_radius_light_days, lag_scaling,
)
from discotau.synthetic import generate_synthetic_dataset
from matplotlib.colors import ListedColormap, LogNorm

from discotau.plotting import wavelength_to_colour, response_function_xlim, blackbody_to_colour

# Purely for the illustrative disk panels -- the fit itself may use bands
# at other wavelengths; this just sets the pictures' colour/size scale.
_DISK_REFERENCE_WAVELENGTH = 5000.0
_DISK_N_R, _DISK_N_PHI = 80, 80
# Floor of the disk panel's colour scale: the zero-torque inner boundary sends
# T -> 0 at the ISCO itself, and below ~1000 K a blackbody barely glows in the
# visible, so extending the scale further would only compress the real range.
_DISK_T_FLOOR_KELVIN = 1000.0


def _draw_observer(ax, x, y, size):
    """A little eye-bearing sphere marking the observer's vantage point."""
    sphere = mpatches.Circle((x, y), size, facecolor="#2b6cb0", edgecolor="0.2", lw=0.6, zorder=5)
    eye_white = mpatches.Ellipse((x, y), size * 1.5, size * 0.8, facecolor="white", edgecolor="0.2", lw=0.6, zorder=6)
    pupil = mpatches.Circle((x, y), size * 0.22, facecolor="k", zorder=7)
    for p in (sphere, eye_white, pupil):
        ax.add_patch(p)


def _expanding_percentiles(samples):
    """68%/95% credible-interval percentiles (plus the median) of
    ``samples`` (shape ``(n_frames, ...)``), computed cumulatively: frame
    ``i``'s percentiles use only ``samples[:i+1]``, not the full run. This
    is what makes the envelope start wide (1-2 samples give almost no
    constraint) and narrow towards the converged posterior's own width as
    the animation progresses, mirroring the credible bands on the standard
    (non-animated) light curve plots -- there computed once, over the
    whole (converged) chain.

    Returns
    -------
    array, shape (5, n_frames, ...)
        The [2.5, 16, 50, 84, 97.5] percentiles, in that order, one set
        per frame.
    """
    n_frames = samples.shape[0]
    out = np.empty((5, n_frames) + samples.shape[1:])
    for i in range(n_frames):
        out[:, i] = np.percentile(samples[: i + 1], [2.5, 16, 50, 84, 97.5], axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-warmup", type=int, default=15, help="Deliberately short -- see module docstring.")
    parser.add_argument("--num-frames", type=int, default=150, help="Post-warmup samples to animate, one per frame.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_dataset(
        bands={"g": 4770.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=40,
        n_freq=20, n_tau=150, tau_max=40.0, noise_level=0.05, seed=0,
    )
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=20, n_tau=150)
    # Diagonal mass matrix on purpose: a dense one (the fit default, CLAUDE.md
    # decision #21) can't adapt in this deliberately tiny warmup, and the
    # point here is to watch an early, still-searching chain anyway.
    ef.fit(
        rng_seed=args.seed, num_warmup=args.num_warmup, num_samples=args.num_frames,
        progress_bar=False, dense_mass=False,
    )

    all_t = np.concatenate([d["t"] for d in ef.bands.values()])
    t_fine = jnp.linspace(all_t.min() - 5.0, all_t.max() + 5.0, 200)
    tau_grid_np = np.asarray(ef.tau_grid)

    S, C = jnp.asarray(ef.samples["S"]), jnp.asarray(ef.samples["C"])
    log_mdot_msun_yr, inclination = jnp.asarray(ef.samples["log_mdot_msun_yr"]), jnp.asarray(ef.samples["inclination"])
    log_mdot_np, inclination_np = np.asarray(log_mdot_msun_yr), np.asarray(inclination)

    def echo_and_psi(S_s, C_s, log_mdot_s, incl_s, wavelength, S_band_s, C_band_s):
        # Via the model module's attribute (not a direct forward_model import)
        # so a swapped model.response_function is honoured here too -- see
        # CLAUDE.md decision #5.
        psi = _model.response_function(ef.tau_grid, log_mdot_msun_yr=log_mdot_s, wavelength=wavelength, inclination=incl_s, M_BH=ef.M_BH)
        A, B = transfer_coeffs(ef.tau_grid, psi, ef.freqs)
        echo = S_band_s * compute_echo(S_s, C_s, ef.freqs, A, B, t_fine) + C_band_s
        return echo, psi

    y_by_band, psi_by_band = {}, {}
    for name, d in ef.bands.items():
        S_band, C_band = jnp.asarray(ef.samples[f"S_{name}"]), jnp.asarray(ef.samples[f"C_{name}"])
        echo_all, psi_all = jax.vmap(
            echo_and_psi, in_axes=(0, 0, 0, 0, None, 0, 0)
        )(S, C, log_mdot_msun_yr, inclination, d["wavelength"], S_band, C_band)
        y_by_band[name] = np.asarray(echo_all)
        psi_by_band[name] = np.asarray(psi_all)
    driver_samples = np.asarray(jax.vmap(lambda S_s, C_s: driver_at(S_s, C_s, ef.freqs, t_fine))(S, C))

    # A single, fixed x-axis limit for every psi(tau) panel, shared with
    # plot_lightcurve_fits (see plotting.response_function_xlim) so the
    # README animation and the standard report/smoke-test output use the
    # same mechanism to set this, not independently-chosen/hardcoded
    # values. Uses the last-fifth of samples (a representative "converged"
    # state), not the whole chain -- early, still-unconverged samples
    # shouldn't set a limit meant to stay fixed for the whole animation.
    tail = max(10, len(next(iter(psi_by_band.values()))) // 5)
    psi_xlim = response_function_xlim(
        tau_grid_np, [psi_by_band[name][-tail:].mean(axis=0) for name in ef.bands]
    )

    # Expanding-window 68%/95% credible envelopes -- wide early (few
    # samples), narrowing towards the converged posterior's own width as
    # the chain accumulates more of them. See _expanding_percentiles.
    driver_pct = _expanding_percentiles(driver_samples)
    echo_pct = {name: _expanding_percentiles(y_by_band[name]) for name in ef.bands}
    psi_pct = {name: _expanding_percentiles(psi_by_band[name]) for name in ef.bands}

    # -- disk-panel geometry (fixed across frames; only colour/tilt vary) --
    r_in = 3.0 * float(_schwarzschild_radius_light_days(ef.M_BH))
    tau_ref_per_sample = np.asarray(lag_scaling(log_mdot_msun_yr, _DISK_REFERENCE_WAVELENGTH, ef.M_BH))
    r_out = 6.0 * float(tau_ref_per_sample.max())
    # Geometric ring edges: resolves the small, hot inner disk (most of the
    # colour change happens within ~tau_ref) without distorting the face-on
    # geometry itself, which stays linear in radius.
    r_grid = np.geomspace(r_in, r_out, _DISK_N_R)
    r_mid = np.sqrt(r_grid[:-1] * r_grid[1:])
    phi_grid = np.linspace(0.0, 2.0 * np.pi, _DISK_N_PHI)
    R_grid, PHI_grid = np.meshgrid(r_grid, phi_grid, indexing="ij")
    X_faceon, Y_faceon = R_grid * np.cos(PHI_grid), R_grid * np.sin(PHI_grid)
    # Each ring coloured by the temperature at its (geometric) midpoint, not
    # its inner edge -- the innermost edge is the ISCO, where T -> 0.
    T_per_r = np.asarray(jax.vmap(
        lambda lm: disk_temperature_profile(r_mid, lm, _DISK_REFERENCE_WAVELENGTH, ef.M_BH)
    )(log_mdot_msun_yr))  # (n_frames, n_r - 1)
    T_grid_all = np.broadcast_to(T_per_r[:, :, None], (len(log_mdot_np), _DISK_N_R - 1, _DISK_N_PHI - 1))

    bof = 2.0 * np.asarray(ef.extra_fields["potential_energy"])

    n_bands = len(ef.bands)
    fig, axes = plt.subplots(
        n_bands + 2, 3, figsize=(13, 1.8 * (n_bands + 2)),
        gridspec_kw={"width_ratios": [3, 1, 1]},
    )

    ax_drv, ax_disk_temp, ax_disk_tilt = axes[0, 0], axes[0, 1], axes[0, 2]
    y_pad = 0.15 * (driver_samples.max() - driver_samples.min())
    ax_drv.set_ylim(driver_samples.min() - y_pad, driver_samples.max() + y_pad)
    ax_drv.set_ylabel("driver X(t)")
    ax_drv.set_title("Driving light curve", fontsize=9)
    ax_drv.grid(alpha=0.6)
    (driver_line,) = ax_drv.plot([], [], color="black", lw=1.5)

    # -- disk-temperature panel: fixed face-on geometry, colour-only updates --
    ax_disk_temp.set_xlim(-1.1 * r_out, 1.1 * r_out)
    ax_disk_temp.set_ylim(-1.1 * r_out, 1.1 * r_out)
    ax_disk_temp.set_aspect("equal")
    ax_disk_temp.set_xticks([])
    ax_disk_temp.set_yticks([])
    disk_title = ax_disk_temp.set_title("", fontsize=8)
    ax_disk_temp.set_facecolor("black")
    # True blackbody colour per temperature, on one log-T scale shared by
    # every frame (see module docstring).
    T_lo = max(float(T_per_r.min()), _DISK_T_FLOOR_KELVIN)
    T_hi = float(T_per_r.max())
    blackbody_cmap = ListedColormap(blackbody_to_colour(np.geomspace(T_lo, T_hi, 256)))
    blackbody_cmap.set_under(blackbody_to_colour(T_lo))
    disk_mesh = ax_disk_temp.pcolormesh(
        X_faceon, Y_faceon, T_grid_all[0], cmap=blackbody_cmap,
        norm=LogNorm(vmin=T_lo, vmax=T_hi), shading="flat",
    )
    cbar = fig.colorbar(disk_mesh, ax=ax_disk_temp, fraction=0.046, pad=0.02)
    cbar.set_label("T (K)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # -- disk-tilt panel: fixed observer/sightline, only the disk line rotates --
    L = r_out
    obs_x = 1.4 * L
    ax_disk_tilt.set_xlim(-1.1 * L, 1.6 * L)
    ax_disk_tilt.set_ylim(-1.1 * L, 1.1 * L)
    ax_disk_tilt.set_aspect("equal")
    ax_disk_tilt.set_xticks([])
    ax_disk_tilt.set_yticks([])
    ax_disk_tilt.set_title("Inclination (observer fixed)", fontsize=8)
    ax_disk_tilt.plot([0.0, obs_x], [0.0, 0.0], "--", color="0.5", lw=0.8, zorder=4)
    _draw_observer(ax_disk_tilt, obs_x, 0.0, 0.09 * L)
    ax_disk_tilt.plot(0.0, 0.0, "o", color="k", ms=4, zorder=4)
    (disk_line,) = ax_disk_tilt.plot([], [], "-", color="#b83227", lw=4, solid_capstyle="round", zorder=3)

    band_lines, psi_lines, band_axes, band_colours = {}, {}, {}, {}
    for row, (name, d) in enumerate(ef.bands.items(), start=1):
        ax_lc, ax_psi = axes[row, 0], axes[row, 1]
        axes[row, 2].axis("off")
        colour = wavelength_to_colour(d["wavelength"])
        band_axes[name] = (ax_lc, ax_psi)
        band_colours[name] = colour
        ax_lc.errorbar(d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=3, color="k", alpha=0.5, zorder=1)
        y = y_by_band[name]
        pad = 0.15 * (y.max() - y.min())
        ax_lc.set_ylim(min(y.min(), d["y"].min()) - pad, max(y.max(), d["y"].max()) + pad)
        ax_lc.set_ylabel(f"{name} ({d['wavelength']:.0f} Å)")
        ax_lc.grid(alpha=0.6)
        (band_lines[name],) = ax_lc.plot([], [], color=colour, lw=1.5, zorder=2)
        ax_lc.sharex(ax_drv)

        psi = psi_by_band[name]
        ax_psi.set_ylim(0.0, 1.05 * psi.max())
        ax_psi.set_xlim(0.0, psi_xlim)
        ax_psi.set_ylabel(f"{name}\nψ(τ)")
        ax_psi.grid(alpha=0.6)
        (psi_lines[name],) = ax_psi.plot([], [], color=colour, lw=1.5)
        if row > 1:
            ax_psi.sharex(axes[1, 1])

    axes[n_bands, 0].set_xlabel("time (days)")
    axes[n_bands, 1].set_xlabel("lag τ (days)")
    ax_drv.set_xlim(float(t_fine.min()), float(t_fine.max()))

    # -- Badness-of-Fit panel: a growing trace, one more point per frame --
    ax_bof, ax_bof_unused1, ax_bof_unused2 = axes[-1, 0], axes[-1, 1], axes[-1, 2]
    ax_bof_unused1.axis("off")
    ax_bof_unused2.axis("off")
    ax_bof.set_xlim(1, args.num_frames)
    bof_pad = 0.05 * (bof.max() - bof.min())
    ax_bof.set_ylim(bof.min() - bof_pad, bof.max() + bof_pad)
    ax_bof.set_xlabel("sample")
    ax_bof.set_ylabel("BOF\n(2 × potential energy)")
    ax_bof.set_title("Badness of Fit", fontsize=9)
    ax_bof.grid(alpha=0.6)
    (bof_line,) = ax_bof.plot([], [], color="0.2", lw=1.2)

    title = fig.suptitle("")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    envelope_artists = []

    def update(i):
        driver_line.set_data(t_fine, driver_samples[i])
        for name in ef.bands:
            band_lines[name].set_data(t_fine, y_by_band[name][i])
            psi_lines[name].set_data(tau_grid_np, psi_by_band[name][i])
        bof_line.set_data(np.arange(1, i + 2), bof[: i + 1])
        title.set_text(f"MCMC sample {i + 1}/{args.num_frames}")

        # 68%/95% credible envelopes, expanding-window over samples seen so
        # far (see _expanding_percentiles) -- fill_between has no in-place
        # update, so these are removed and redrawn each frame.
        for artist in envelope_artists:
            artist.remove()
        envelope_artists.clear()
        envelope_artists.append(ax_drv.fill_between(t_fine, driver_pct[0, i], driver_pct[4, i], color="0.5", alpha=0.15, zorder=0))
        envelope_artists.append(ax_drv.fill_between(t_fine, driver_pct[1, i], driver_pct[3, i], color="0.5", alpha=0.3, zorder=0))
        for name in ef.bands:
            ax_lc, ax_psi = band_axes[name]
            colour = band_colours[name]
            envelope_artists.append(ax_lc.fill_between(t_fine, echo_pct[name][0, i], echo_pct[name][4, i], color=colour, alpha=0.15, zorder=0))
            envelope_artists.append(ax_lc.fill_between(t_fine, echo_pct[name][1, i], echo_pct[name][3, i], color=colour, alpha=0.3, zorder=0))
            envelope_artists.append(ax_psi.fill_between(tau_grid_np, psi_pct[name][0, i], psi_pct[name][4, i], color=colour, alpha=0.15, zorder=0))
            envelope_artists.append(ax_psi.fill_between(tau_grid_np, psi_pct[name][1, i], psi_pct[name][3, i], color=colour, alpha=0.3, zorder=0))

        disk_mesh.set_array(T_grid_all[i].ravel())
        disk_title.set_text(f"log_mdot_msun_yr = {log_mdot_np[i]:.2f}, inclination = {inclination_np[i]:.1f}°")

        incl_rad = np.deg2rad(float(inclination_np[i]))
        dx, dy = L * np.sin(incl_rad), L * np.cos(incl_rad)
        disk_line.set_data([-dx, dx], [-dy, dy])

        return [driver_line, title, disk_mesh, disk_title, disk_line, bof_line,
                *band_lines.values(), *psi_lines.values(), *envelope_artists]

    ani = FuncAnimation(fig, update, frames=args.num_frames, blit=False)
    out_path = args.outdir / "fit_animation.gif"
    ani.save(out_path, writer="pillow", fps=args.fps, dpi=args.dpi)
    plt.close(fig)

    # Re-palette to shrink the file -- a full-colour GIF from matplotlib is
    # far larger than this plot needs.
    from PIL import Image, ImageSequence

    src = Image.open(out_path)
    frames = [f.convert("RGB").quantize(colors=96, method=Image.MEDIANCUT) for f in ImageSequence.Iterator(src)]
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
