"""
smoke_test.py
=============

Quick "did I break anything" sanity checks, not a full numeric pass/fail
test (see tests/test_recovery.py for that) -- run these by eye.

Two datasets:

- **synthetic** (default): generate a synthetic multi-band dataset, run a
  short NUTS fit (enough to see whether the model is doing something
  sensible, not enough to fully converge), and save plots of the raw
  data, the posterior-predictive fit, and MCMC trace diagnostics. Run this
  after touching forward_model.py / model.py / echofit.py.
- **ngc5548**: the same short-fit-and-plot check, but on a small subset of
  the real NGC 5548 AGN STORM data (downloaded via
  scripts/download_ngc5548_storm_data.py) instead of synthetic data --
  a fast pre-flight check that real data loads and fits without anything
  obviously wrong (crashes, NaNs, a wild divergence rate) before
  committing to scripts/run_ngc5548_fit.sh's much longer full run. There's
  no ground truth to compare against here, unlike the synthetic check.

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --num-warmup 500 --num-samples 500 --outdir /tmp/smoke
    python scripts/smoke_test.py --dataset ngc5548
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from discotau.synthetic import generate_synthetic_dataset
from discotau.echofit import EchoFit
from discotau import reporting

sys.path.insert(0, str(Path(__file__).parent))
import download_ngc5548_storm_data as ngc5548_dl  # noqa: E402 (needs sys.path set first)

# 4 bands spanning the full wavelength range -- enough to sanity-check that
# the real data loads and fits without paying the full 13-band run's cost.
# Same subset as run_ngc5548_fit.sh's own "quick look" step.
NGC5548_SMOKE_BANDS = ["uv1158", "u", "g", "z"]


def _run_synthetic(args):
    # Same well-resolved-lag setup as tests/test_recovery.py: M_BH large
    # enough that mean lags (days) are well above the sampling cadence, so
    # the echo shape is actually visible by eye rather than buried in noise.
    # Two observing gaps make the campaign more realistic (and harder for
    # NUTS/the frequency-grid estimate) than fully uniform random sampling.
    gaps = None if args.no_gaps else [(50, 14), (150, 21)]
    data = generate_synthetic_dataset(
        M_BH=1.0e9,
        log_mdot_true=0.0,
        inclination_true=35.0,
        sigma_drw_true=0.3,
        tau_drw_true=30.0,
        bands={"g": 4770.0, "i": 7625.0},
        t_span=200.0,
        n_obs_per_band=50,
        n_freq=15,
        n_tau=100,
        tau_max=50.0,
        noise_level=args.noise_level,
        gaps=gaps,
        seed=args.seed,
    )
    truth = data["truth"]

    ef = EchoFit(M_BH=truth["M_BH"])
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=15, n_tau=100)

    print(f"Fitting synthetic data ({args.num_warmup} warmup + {args.num_samples} samples, 1 chain)...")
    t0 = time.time()
    ef.fit(
        rng_seed=0, num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=1, progress_bar=False, dense_mass=not args.diagonal_mass,
    )
    dt = time.time() - t0

    diverging = np.asarray(ef.extra_fields.get("diverging", []))
    n_div = int(diverging.sum()) if diverging.size else 0
    log_mdot_mean = float(ef.samples["log_mdot"].mean())

    print(f"Fit finished in {dt:.1f}s")
    print(f"Divergent transitions: {n_div}/{len(diverging)}")
    print(f"log_mdot posterior mean: {log_mdot_mean:.3f} (truth: {truth['log_mdot']:.3f})")

    print("Rendering plots...")
    report_path = reporting.generate_report(ef, args.outdir, fit_seconds=dt, truth=truth)

    print(f"\nSaved plots to {args.outdir}/")
    print(f"Open {report_path} to view everything in one page.")


def _run_ngc5548(args):
    raw_dir = args.outdir / "ngc5548_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table3_path = ngc5548_dl._download("table3.dat", raw_dir)
    table4_path = ngc5548_dl._download("table4.dat", raw_dir)

    optical = ngc5548_dl._parse_table3(table3_path)
    uv = ngc5548_dl._parse_table4(table4_path)
    all_bands = {name: (wl, optical[name]) for name, wl in ngc5548_dl.OPTICAL_WAVELENGTHS.items()}
    all_bands.update({name: (lam, uv[name]) for lam, name in ngc5548_dl.UV_BAND_NAMES.items()})

    t0_global = min(min(d["t"]) for _, d in all_bands.values())
    m_bh = args.m_bh if args.m_bh is not None else ngc5548_dl.M_BH_STARKEY_STORM_VI

    ef = EchoFit(M_BH=m_bh)
    for name in NGC5548_SMOKE_BANDS:
        wavelength, d = all_bands[name]
        t = np.asarray(d["t"]) - t0_global
        ef.add_lightcurve(name, wavelength=wavelength, t=t, y=np.asarray(d["y"]), yerr=np.asarray(d["yerr"]))
    ef.build_grid()

    print(f"Fitting real NGC 5548 data, bands {NGC5548_SMOKE_BANDS} "
          f"({args.num_warmup} warmup + {args.num_samples} samples, M_BH={m_bh:.3g})...")
    t0 = time.time()
    ef.fit(
        rng_seed=0, num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=1, progress_bar=False, dense_mass=not args.diagonal_mass,
    )
    dt = time.time() - t0

    diverging = np.asarray(ef.extra_fields.get("diverging", []))
    n_div = int(diverging.sum())
    div_frac = n_div / len(diverging) if len(diverging) else 0.0
    all_finite = all(np.all(np.isfinite(np.asarray(v))) for v in ef.samples.values())

    print(f"Fit finished in {dt:.1f}s")
    print(f"Divergent transitions: {n_div}/{len(diverging)} ({div_frac:.0%})")
    print(f"All sample values finite: {all_finite}")
    print(f"\n{'parameter':<14}{'mean':>10}{'std':>10}")
    # tau_drw only exists as a site if this fit used drw_prior=True (see
    # CLAUDE.md decision #19); the default random-walk prior has no such site.
    for name in ["log_mdot", "inclination", "sigma_drw", "tau_drw"]:
        if name not in ef.samples:
            continue
        s = np.asarray(ef.samples[name])
        print(f"{name:<14}{s.mean():>10.3f}{s.std():>10.3f}")

    if not all_finite:
        print("\n*** WARNING: non-finite values in posterior samples -- something is wrong. ***")
    if div_frac > 0.3:
        print(f"\n*** WARNING: {div_frac:.0%} divergent transitions -- that's high even for a short, "
              f"un-tuned smoke-test chain; investigate before trusting a longer real run. ***")

    print("\nRendering plots...")
    report_path = reporting.generate_report(ef, args.outdir, fit_seconds=dt, title="ngc5548_smoke_test")

    print(f"\nSaved plots to {args.outdir}/")
    print(f"Open {report_path} to view everything in one page.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("synthetic", "ngc5548"), default="synthetic")
    parser.add_argument("--num-warmup", type=int, default=None, help="Default: 300 (synthetic) / 30 (ngc5548).")
    parser.add_argument("--num-samples", type=int, default=None, help="Default: 300 (synthetic) / 30 (ngc5548).")
    parser.add_argument("--seed", type=int, default=2, help="Synthetic data only.")
    parser.add_argument(
        "--noise-level", type=float, default=0.08,
        help=(
            "Synthetic data only. Deliberately higher than tests/test_recovery.py's "
            "0.02 (tuned for a tight numeric check) so the posterior-predictive "
            "credible bands are actually visible by eye here -- at 0.02 the 95%% CI "
            "is ~1%% of the data's y-range, i.e. a couple of pixels."
        ),
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Directory to write PNGs + report.html into. Defaults to "
             "smoke_test_output/ (synthetic) or ngc5548_smoke_test_output/ "
             "(ngc5548) -- kept separate so running one doesn't silently "
             "overwrite the other's committed example output.",
    )
    parser.add_argument(
        "--no-gaps", action="store_true",
        help="Synthetic data only. Disable the default observing gaps (2 weeks from day 50, 3 weeks from day 150).",
    )
    parser.add_argument(
        "--m-bh", type=float, default=None,
        help="ngc5548 dataset only. Defaults to the Starkey et al. 2017 (STORM Paper VI) value; see README.md.",
    )
    parser.add_argument(
        "--diagonal-mass", action="store_true",
        help="Use a diagonal NUTS mass matrix instead of EchoFit.fit()'s dense default -- see CLAUDE.md decisions #17/#21.",
    )
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = Path("smoke_test_output" if args.dataset == "synthetic" else "ngc5548_smoke_test_output")
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "synthetic":
        if args.num_warmup is None:
            args.num_warmup = 300
        if args.num_samples is None:
            args.num_samples = 300
        _run_synthetic(args)
    else:
        if args.num_warmup is None:
            args.num_warmup = 30
        if args.num_samples is None:
            args.num_samples = 30
        _run_ngc5548(args)


if __name__ == "__main__":
    main()
