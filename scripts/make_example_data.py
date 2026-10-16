"""
make_example_data.py
=====================

Writes a synthetic two-band light curve dataset to plain text files, in the
``t y yerr`` format ``scripts/fit_lightcurves.py`` reads -- so there's
something to point that script at without needing real data first. See
``scripts/run_example_fit.sh`` for the full worked example this feeds into.

Usage
-----
    python scripts/make_example_data.py --outdir example_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from discotau.synthetic import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("example_data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_dataset(
        bands={"g": 4770.0, "i": 7625.0},
        M_BH=1.0e8,
        n_obs_per_band=60,
        noise_level=0.05,
        gaps=[(50.0, 14.0), (150.0, 21.0)],
        seed=args.seed,
    )

    for name, d in data["bands"].items():
        path = args.outdir / f"{name}_band.txt"
        np.savetxt(
            path, np.column_stack([d["t"], d["y"], d["yerr"]]),
            header=f"t(days) y yerr -- band {name!r}, wavelength {d['wavelength']:.0f} A", fmt="%.6f",
        )
        print(f"wrote {path} ({len(d['t'])} points)")

    truth = data["truth"]
    print(
        f"\nGround truth: M_BH={truth['M_BH']:.3g}, log_mdot={truth['log_mdot']:.3g}, "
        f"inclination={truth['inclination']:.3g} deg"
    )


if __name__ == "__main__":
    main()
