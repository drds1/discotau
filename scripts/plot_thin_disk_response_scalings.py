"""
plot_thin_disk_response_scalings.py
====================================

Generates the three figures used in docs/thin_disk_response.md: a
case-study accretion disk's response function
``forward_model.thin_disk_response`` evaluated (a) at fixed accretion rate
for a range of inclinations, and (b) face-on for a range of accretion
rates, each with a vertical line at the response's own (numerically
integrated) mean lag; and (c) a comparison of the exact response against
the fast, precomputed-template approximation from
``forward_model.build_thin_disk_response_fast``.

This is a documentation/figure-generation script, not a test -- run it
directly to regenerate the PNGs under docs/images/ if thin_disk_response's
physics or default parameters ever change:

    poetry run python scripts/plot_thin_disk_response_scalings.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from discotau.forward_model import lag_scaling, thin_disk_response, build_thin_disk_response_fast, typical_log_mdot

_np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

OUT_DIR = "docs/images"

# Case-study disk: same M_BH/wavelength pivot used throughout the test
# suite, so the lags below line up with what CLAUDE.md/README already
# describe.
M_BH = 1.0e8       # solar masses
WAVELENGTH = 5000.0  # Angstrom


def _mean_lag(tau_grid, psi):
    """Numerically integrated mean lag of a response, int psi*tau dtau."""
    return float(_np_trapz(np.asarray(psi) * np.asarray(tau_grid), np.asarray(tau_grid)))


def plot_inclination_sweep():
    """Fixed accretion rate, varying inclination -- the mean lag should be
    (numerically) the same for every curve, since the disk's iso-delay
    surface reshapes the response's skew without changing its mean:
    int_0^2pi (1 + sin(i) cos(phi)) dphi = 2*pi for any inclination i, the
    cos(phi) term integrating to zero. Inclination changes the *shape*, not
    the *mean*."""
    tau_grid = np.linspace(-2.0, 15.0, 6000)
    inclinations = [0.0, 20.0, 40.0, 60.0, 80.0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = plt.cm.viridis(np.linspace(0.15, 0.9, len(inclinations)))
    for inclination, colour in zip(inclinations, colours):
        psi = thin_disk_response(
            tau_grid, log_mdot_msun_yr=float(typical_log_mdot(M_BH)), wavelength=WAVELENGTH,
            inclination=inclination, M_BH=M_BH,
        )
        mean_lag = _mean_lag(tau_grid, psi)
        print(f"inclination {inclination:.0f}: mean lag {mean_lag:.3f} d")
        ax.plot(tau_grid, psi, color=colour, label=f"inclination={inclination:.0f} deg")
        ax.axvline(mean_lag, color=colour, linestyle="--", linewidth=1, alpha=0.8)

    ax.set_xlim(-0.2, 4.0)  # the responses are compact; the grid runs further for the means
    ax.set_xlabel("lag, tau (days)")
    ax.set_ylabel("psi(tau)")
    ax.set_title(
        f"Fixed Mdot = {10 ** float(typical_log_mdot(M_BH)):.3f} Msun/yr (M_BH={M_BH:.0e} Msun, lambda={WAVELENGTH:.0f} A),\n"
        "varying inclination -- dashed lines are each curve's own mean lag",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/thin_disk_response_inclination_sweep.png", dpi=150)
    plt.close(fig)


def plot_mdot_sweep():
    """Face-on (inclination=0), varying accretion rate -- the mean lag
    should scale as mdot**(1/3) (lag_scaling's own scaling law, at fixed
    M_BH and wavelength), which is exactly what the vertical lines are
    here to make visible."""
    tau_grid = np.linspace(-2.0, 12.0, 6000)
    log_mdots = [float(typical_log_mdot(M_BH)) + x for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(log_mdots)))
    for log_mdot_msun_yr, colour in zip(log_mdots, colours):
        psi = thin_disk_response(
            tau_grid, log_mdot_msun_yr=log_mdot_msun_yr, wavelength=WAVELENGTH,
            inclination=0.0, M_BH=M_BH,
        )
        mean_lag = _mean_lag(tau_grid, psi)
        print(f"log_mdot_msun_yr {log_mdot_msun_yr:+.1f}: mean lag {mean_lag:.3f} d, lag_scaling {float(lag_scaling(log_mdot_msun_yr, WAVELENGTH, M_BH)):.3f} d")
        mdot = 10.0 ** log_mdot_msun_yr
        ax.plot(tau_grid, psi, color=colour, label=f"Mdot = {mdot:.4f} Msun/yr")
        ax.axvline(mean_lag, color=colour, linestyle="--", linewidth=1, alpha=0.8)

    ax.set_xlabel("lag, tau (days)")
    ax.set_ylabel("psi(tau)")
    ax.set_title(
        f"Face-on (inclination=0 deg, M_BH={M_BH:.0e} Msun, lambda={WAVELENGTH:.0f} A),\n"
        "varying mdot -- dashed lines are each curve's own mean lag, scaling as mdot**(1/3)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/thin_disk_response_mdot_sweep.png", dpi=150)
    plt.close(fig)


def plot_fast_vs_slow():
    """Compares the exact disk integral against build_thin_disk_response_fast's
    precomputed-template-plus-stretch approximation: a case close to the
    table's own reference point (where the stretch is small), and a
    deliberately harder case far from it (high inclination -- the response's
    sharpest feature -- at a different wavelength) to show honestly where
    the self-similar-stretching approximation starts to strain."""
    fast = build_thin_disk_response_fast(M_BH)
    tau_grid = np.linspace(-2.0, 15.0, 600)
    cases = [
        {"log_mdot_msun_yr": float(typical_log_mdot(M_BH)), "wavelength": WAVELENGTH, "inclination": 30.0,
         "label": "near the table's reference point", "zoom": None},
        {"log_mdot_msun_yr": float(typical_log_mdot(M_BH)) - 0.5, "wavelength": 7000.0, "inclination": 85.0,
         "label": "far from it: high inclination, different wavelength", "zoom": (-0.1, 0.6)},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, case in zip(axes, cases):
        psi_slow = np.asarray(thin_disk_response(
            tau_grid, log_mdot_msun_yr=case["log_mdot_msun_yr"], wavelength=case["wavelength"],
            inclination=case["inclination"], M_BH=M_BH,
        ))
        psi_fast = np.asarray(fast(tau_grid, case["log_mdot_msun_yr"], case["wavelength"], case["inclination"], M_BH))
        ax.plot(tau_grid, psi_slow, label="exact (thin_disk_response)")
        ax.plot(tau_grid, psi_fast, "--", label="fast (templated + stretched)")
        title = (
            f"log_mdot_msun_yr={case['log_mdot_msun_yr']}, wavelength={case['wavelength']:.0f} A, "
            f"inclination={case['inclination']:.0f} deg\n{case['label']}"
        )
        if case["zoom"] is not None:
            title += " -- zoomed to the peak"
            ax.set_xlim(*case["zoom"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("lag, tau (days)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("psi(tau)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/thin_disk_response_fast_vs_slow.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_inclination_sweep()
    plot_mdot_sweep()
    plot_fast_vs_slow()
    print(f"Wrote figures to {OUT_DIR}/")
