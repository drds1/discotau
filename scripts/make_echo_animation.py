"""
make_echo_animation.py
======================

The README header animation: a flickering lamppost echoed by an accretion disc.

What pycream2 models, drawn directly: a compact source above the black hole (the
lamppost) varies; its light heats the disc after a light-travel delay that
depends on where each element sits, tau = sqrt(r^2 + h^2) + h cos i + y sin i
(the near side, y < 0, first); and every band's light curve is the lamppost's
light curve smeared and delayed by that band's response function.

- The disc: each pixel's temperature is the viscous profile plus the lamppost's
  heating at the delayed time, T^4 = T_visc^4 + T_irr^4 L(t - tau)/<L>, with
  T_irr^4 ~ h/d^3, drawn in its true blackbody colour
  (pycream2.plotting.blackbody_to_colour), so every flicker sweeps outwards as
  a ripple that reaches the near side first.
- The light curves: the lamppost's L(t) (white) and each band's echo, L
  convolved with pycream2.forward_model.thin_disk_response at the same
  accretion rate, inclination and mass, in its wavelength's colour
  (pycream2.plotting.wavelength_to_colour), with the responses alongside. All
  are drawn on the lamppost's scale, so the longer wavelengths' smaller, smoother
  and later variations are visible as such.
- The lamppost varies stochastically, as an AGN does: red noise with a bending
  power-law spectrum (~1/f below the bend, ~1/f^2 above), log-normal in flux.

No code name is drawn into the frames, so a rename doesn't need a new render.

The physics is the code's; only the variability amplitude is exaggerated, so
that the ripples are visible. No fitting is done.

Usage (from the repository root; a few minutes, no fitting):
    python scripts/make_echo_animation.py                      # docs/images/echo_animation.webp
    python scripts/make_echo_animation.py --out echo.gif       # a GIF instead (larger)
    python scripts/make_echo_animation.py --still 120 --out still.png   # one frame, to check the look

An animated WebP needs a Pillow built with WebP animation support (as on
Kaggle and most Linux wheels).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from pycream2.forward_model import disk_temperature_profile, thin_disk_response
from pycream2.plotting import blackbody_to_colour, wavelength_to_colour

# The disc: Fairall 9's black-hole mass, with an accretion rate that runs the
# drawn disc from ~30,000 K inside to ~2,800 K at its rim (blue-white to
# orange-red), viewed at a dramatic inclination.
M_BH = 2.55e8  # Msun
LOG_MDOT = 0.5
INCLINATION = 62.0  # degrees
RS_LD = 2 * 6.674e-8 * M_BH * 1.989e33 / 2.998e10 ** 2 / (2.998e10 * 86400)  # Schwarzschild radius, light-days
R_IN, H_LAMP = 3 * RS_LD, 3 * RS_LD  # inner edge and lamppost height, light-days
R_OUT = 7.0  # light-days, edge of the drawn disc
IRRADIATION_REF = 1.0  # light-days: the lamppost's mean heating equals the viscous at this radius
# Bands: rest wavelength (Angstrom) and label.
BANDS = [(1500, "FUV"), (4400, "B"), (5500, "V"), (8000, "I")]

# The animation.
T_START, T_END = 30.0, 80.0  # days shown (the driver starts earlier, so the echoes are settled)
N_FRAMES = 200
WINDOW = 16.0  # days of light curve on screen
TAU_MAX = 30.0  # response grid, days
DT = 0.05  # driver grid, days
SEED = 3
F_BEND, F_CUT = 0.1, 0.8  # cycles/day: the spectrum's bend and a smooth cut-off at the shortest timescales
LOG_SIGMA = 0.6  # rms of ln L
ROW_AMPLITUDE = 0.4  # light-curve rows: one lamppost standard deviation, in row heights

# Layout, in pixels at DPI.
WIDTH, HEIGHT, DPI = 880, 640, 100
DISC_FRACTION = 0.6  # top part of the figure for the disc
BACKGROUND = "#03030a"
BACKGROUND_RGB = np.array([3, 3, 10]) / 255.0
SATURATION = 2.1


def driver(seed=SEED):
    """The lamppost's light curve L(t) > 0 on a fine grid: red noise with a
    bending power-law spectrum, P(f) ~ 1 / (f (1 + f / F_BEND)), as AGN X-ray
    light curves show, exponentiated so the flux is log-normal (flares are sharper
    than dips, as in real light curves)."""
    rng = np.random.default_rng(seed)
    t = np.arange(-TAU_MAX - 5.0, T_END + 1.0, DT)
    freqs = np.fft.rfftfreq(len(t), DT)[1:]
    power = 1.0 / (freqs * (1.0 + freqs / F_BEND)) * np.exp(-freqs / F_CUT)
    coeffs = np.sqrt(power) * (rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs)))
    walk = np.fft.irfft(np.concatenate([[0.0], coeffs]), n=len(t))
    walk = (walk - walk.mean()) / walk.std()
    light = np.exp(LOG_SIGMA * walk)
    return t, light / light.mean()


def echoes(t, light):
    """Each band's response and its light curve, L convolved with psi."""
    tau = np.arange(0.0, TAU_MAX, DT)
    out = {}
    for wl, label in BANDS:
        psi = np.asarray(thin_disk_response(tau, LOG_MDOT, float(wl), INCLINATION, M_BH), dtype=float)
        psi /= psi.sum()
        out[label] = dict(wavelength=wl, psi=psi, curve=np.convolve(light, psi)[: len(t)])
    return tau, out


class DiscRenderer:
    """The disc as seen by the observer (orthographic, inclined about the x axis),
    rasterised to an RGB image with a bloom."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        cos_i, sin_i = np.cos(np.deg2rad(INCLINATION)), np.sin(np.deg2rad(INCLINATION))
        scale = 0.82 * (width / 2) / R_OUT  # pixels per light-day
        x_pix, y_pix = np.meshgrid(np.arange(width) - width / 2, height * 0.5 - np.arange(height))
        x, y = x_pix / scale, y_pix / scale / cos_i  # back onto the disc plane
        self.r = np.hypot(x, y)
        self.on_disc = (self.r > R_IN) & (self.r < R_OUT)
        r = np.clip(self.r, R_IN * 1.001, None)
        d = np.sqrt(r ** 2 + H_LAMP ** 2)
        # Sky y increases upwards and the near side (disc y < 0) is at the bottom.
        self.tau = d + H_LAMP * cos_i + y * sin_i
        t4_visc = np.asarray(disk_temperature_profile(r, LOG_MDOT, 5000.0, M_BH), dtype=float) ** 4
        irr = H_LAMP / d ** 3
        d_ref = np.sqrt(IRRADIATION_REF ** 2 + H_LAMP ** 2)
        t4_ref = float(np.asarray(disk_temperature_profile(np.array([IRRADIATION_REF]), LOG_MDOT, 5000.0, M_BH))[0]) ** 4
        self.t4_visc, self.t4_irr = t4_visc, irr * t4_ref / (H_LAMP / d_ref ** 3)
        # Soft edges, so the disc fades out rather than ending on a hard line.
        self.edge = np.clip((R_OUT - self.r) / (0.25 * R_OUT), 0, 1) ** 1.5 * self.on_disc
        self.lamp_xy = (width / 2, height * 0.5 - H_LAMP * sin_i * scale)
        # Colour and brightness looked up on a log-temperature grid (fast).
        self.log_t = np.linspace(np.log10(1500.0), np.log10(2e5), 512)
        self.rgb_table = blackbody_to_colour(10 ** self.log_t)
        # A starfield, with a slow twinkle.
        rng = np.random.default_rng(11)
        n = 420
        self.stars = (rng.uniform(0, width, n), rng.uniform(0, height, n), rng.uniform(0.15, 1.0, n) ** 3,
                      rng.uniform(0, 2 * np.pi, n), rng.uniform(0.5, 2.0, n))
        self.t_ref = 4500.0  # brightness scale (Kelvin): Reinhard tone mapping of T^4 about it

    def render(self, t, t_grid, light):
        level = np.interp(t - self.tau, t_grid, light)
        t4 = self.t4_visc + self.t4_irr * level
        temp = t4 ** 0.25
        log_t = np.clip(np.log10(temp), self.log_t[0], self.log_t[-1])
        idx = np.clip(((log_t - self.log_t[0]) / (self.log_t[-1] - self.log_t[0]) * 511).astype(int), 0, 511)
        rgb = self.rgb_table[idx]
        grey = rgb.mean(axis=-1, keepdims=True)
        rgb = np.clip(grey + SATURATION * (rgb - grey), 0, 1)  # blackbody colours are pale: richer for display
        lum = (temp / self.t_ref) ** 4
        bright = (lum / (1.0 + lum)) ** 0.5
        img = rgb * bright[..., None] * self.edge[..., None]
        # The lamppost: a point whose brightness follows L(t).
        lamp = np.zeros(self.r.shape)
        lx, ly = self.lamp_xy
        iy, ix = int(round(ly)), int(round(lx))
        lamp[iy, ix] = 60.0 * np.interp(t, t_grid, light)
        img += gaussian_filter(lamp, 2.0)[..., None] * np.array([0.85, 0.9, 1.0])
        img += gaussian_filter(lamp, 9.0)[..., None] * np.array([0.6, 0.7, 1.0]) * 4.0
        # Bloom: a blurred copy of the bright parts added back.
        glow = gaussian_filter(np.clip(img - 0.6, 0, None), sigma=(5, 5, 0))
        img = img + 0.5 * glow
        # Stars behind (hidden by the disc where it is bright).
        sx, sy, sb, sp, sf = self.stars
        twinkle = sb * (0.75 + 0.25 * np.sin(sp + sf * t))
        star = np.zeros(self.r.shape)
        star[sy.astype(int), sx.astype(int)] = twinkle
        star = gaussian_filter(star, 0.6) * 4.0
        img += (star * (1 - np.clip(self.edge * 3, 0, 1)))[..., None] * np.array([0.85, 0.9, 1.0])
        return np.clip(img + BACKGROUND_RGB, 0, 1)


def frame(fig_parts, t, t_grid, light, tau, bands):
    fig, ax_img, image, ax_lc, lines, heads, now_line, ax_psi, renderer = fig_parts
    image.set_data(renderer.render(t, t_grid, light))
    lo = t - WINDOW
    shown = (t_grid >= lo) & (t_grid <= t)
    scale = ROW_AMPLITUDE / np.std(light[t_grid >= T_START])
    for k, (key, line) in enumerate(lines.items()):
        y = light if key == "lamppost" else bands[key]["curve"]
        y_norm = (y - 1.0) * scale  # every echo has the lamppost's mean (psi is normalised)
        offset = row_offset(k, len(lines))
        line.set_data(t_grid[shown] - t, y_norm[shown] + offset)
        heads[key].set_data([0.0], [np.interp(t, t_grid, y_norm) + offset])
    return fig


def row_offset(k, n):
    """Baseline of light-curve row k of n (the lamppost on top, with extra room for its flares)."""
    return (n - 1 - k) * 1.0 + (0.5 if k == 0 else 0.0)


def build_figure(renderer, tau, bands):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#c9cbe0", "axes.edgecolor": "#3a3c55",
                         "xtick.color": "#8a8ca8", "ytick.color": "#8a8ca8"})
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BACKGROUND)
    ax_img = fig.add_axes([0, 1 - DISC_FRACTION, 1, DISC_FRACTION])
    ax_img.set_axis_off()
    image = ax_img.imshow(np.zeros((renderer.height, renderer.width, 3)), interpolation="bilinear")
    fig.text(0.025, 0.95, "a flickering lamppost, echoed by an accretion disc", fontsize=11, color="#b4b6d8")
    fig.text(0.975, 0.955, f"inclination {INCLINATION:.0f}\u00b0, light-travel delays across the disc",
             fontsize=8, color="#7a7ca0", ha="right")
    # The light curves (left) and responses (right).
    ax_lc = fig.add_axes([0.07, 0.07, 0.62, 1 - DISC_FRACTION - 0.1], facecolor=BACKGROUND)
    keys = ["lamppost"] + [label for _, label in BANDS]
    colours = {"lamppost": "#ffffff", **{label: wavelength_to_colour(wl) for wl, label in BANDS}}
    lines, heads = {}, {}
    for k, key in enumerate(keys):
        c = colours[key]
        offset = row_offset(k, len(keys))
        ax_lc.plot([-WINDOW, 0], [offset, offset], color="#1a1c2c", lw=0.6, zorder=0)
        (lines[key],) = ax_lc.plot([], [], color=c, lw=1.3, solid_capstyle="round")
        (heads[key],) = ax_lc.plot([], [], "o", color=c, ms=5, mec="white", mew=0.6)
        name = "lamppost" if key == "lamppost" else f"{key}  {dict((b[1], b[0]) for b in BANDS)[key]} Å"
        ax_lc.text(0.35, offset + 0.4, name, color=c, fontsize=7.5, va="center")
    ax_lc.set_xlim(-WINDOW, 2.6)
    ax_lc.set_ylim(-0.7, len(keys) + 1.1)
    ax_lc.set_yticks([])
    ax_lc.set_xlabel("days before now", fontsize=8, color="#8a8ca8")
    ax_lc.tick_params(labelsize=7)
    for side in ("top", "right", "left"):
        ax_lc.spines[side].set_visible(False)
    now_line = ax_lc.axvline(0, color="#5a5c80", lw=0.7, ls=":")
    ax_psi = fig.add_axes([0.74, 0.07, 0.23, 1 - DISC_FRACTION - 0.1], facecolor=BACKGROUND)
    for wl, label in BANDS:
        psi = bands[label]["psi"] / bands[label]["psi"].max()
        ax_psi.plot(tau, psi, color=colours[label], lw=1.2)
    ax_psi.set_xlim(0, 12)
    ax_psi.set_ylim(0, 1.08)
    ax_psi.set_yticks([])
    ax_psi.set_xlabel(r"delay $\tau$ (days)", fontsize=8, color="#8a8ca8")
    ax_psi.set_title(r"response $\psi(\tau)$", fontsize=8, color="#9a9cc0", loc="left")
    ax_psi.tick_params(labelsize=7)
    for side in ("top", "right", "left"):
        ax_psi.spines[side].set_visible(False)
    return fig, ax_img, image, ax_lc, lines, heads, now_line, ax_psi, renderer


def to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BACKGROUND)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--still", type=int, help="render only this frame number, as a PNG")
    parser.add_argument("--out", default="docs/images/echo_animation.webp")
    parser.add_argument("--colours", type=int, default=256, help="GIF palette size")
    args = parser.parse_args()
    t_grid, light = driver()
    tau, bands = echoes(t_grid, light)
    renderer = DiscRenderer(WIDTH, int(HEIGHT * DISC_FRACTION))
    parts = build_figure(renderer, tau, bands)
    times = np.linspace(T_START, T_END, N_FRAMES)
    if args.still is not None:
        frame(parts, times[args.still], t_grid, light, tau, bands)
        to_image(parts[0]).save(args.out)
        print(f"wrote {args.out} (t = {times[args.still]:.1f} d)")
        return
    frames = [to_image(frame(parts, t, t_grid, light, tau, bands)) for t in times]
    out = Path(args.out)
    if out.suffix == ".webp":
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=60, loop=0, quality=80, method=6)
    else:
        # One palette from a montage of many frames, so every colour that appears
        # anywhere is in it (a single frame's palette loses the violet UV line).
        width, height = frames[0].size
        picks = frames[::20]
        montage = Image.new("RGB", (width, height * len(picks)))
        for k, f in enumerate(picks):
            montage.paste(f, (0, k * height))
        palette = montage.quantize(colors=args.colours, method=Image.Quantize.MEDIANCUT)
        quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        quantised[0].save(out, save_all=True, append_images=quantised[1:], duration=60, loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
