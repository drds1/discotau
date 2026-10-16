#!/usr/bin/env bash
# run_ngc5548_fit.sh
# ===================
#
# A worked, runnable example of discotau applied to a *real* AGN light
# curve dataset: NGC 5548's AGN STORM continuum monitoring campaign
# (Fausnaugh, Denney, Barth, et al. 2016, ApJ 821, 56), downloaded
# directly from VizieR (catalog J/ApJ/821/56). See README.md's
# "Real-data worked example" section for the full story.
#
# Real data is slower than the synthetic examples elsewhere in this repo:
# 13 bands, ~2600 data points total, real (irregular, gappy) cadence, from
# 16 different observatories. Step 2 below is a fast 4-band look to check
# the pipeline end to end before committing to the full 13-band run in
# Step 3, which takes about 1.5 to 2 hours on a laptop CPU.
#
# Both steps pass --fit-error-model. Without it the full set does not fit:
# measured on all 13 bands, dense-mass NUTS hit the 1023-step tree-depth
# ceiling on every sample with 86/300 divergences and a minimum ESS of 4/300,
# and .optimise() stopped at a non-positive-definite Hessian. With it: 2/300
# divergences, minimum ESS 32/300, and NUTS and .optimise() agree on log_mdot
# (1.62 +/- 0.05 vs 1.63). The quoted errors are fine point to point, but at
# 0.2-0.7% of the flux they are far smaller than the thin-disk model's own
# mismatch; the fitted error model inflates the far-UV errors ~3x and adds
# jitter of 1-3x the quoted error to every band.
#
# Usage:
#   ./scripts/run_ngc5548_fit.sh
#
# Run from the repository root. If you're not using `poetry shell`, prefix
# every `python` call below with `poetry run` instead (already done here).

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_PY="poetry run python"

# ---------------------------------------------------------------------
# Step 1: download the real light curve data (VizieR J/ApJ/821/56). See
# scripts/download_ngc5548_storm_data.py's own docstring for exactly what
# this fetches and how the per-band files are built.
# ---------------------------------------------------------------------
echo "== Step 1: downloading NGC 5548 AGN STORM light curves =="
$RUN_PY scripts/download_ngc5548_storm_data.py --outdir ngc5548_storm_data
ls -l ngc5548_storm_data/

# ---------------------------------------------------------------------
# Step 2: a quick look first -- 4 bands spanning the wavelength range
# (UV, u, g, z), a purely in-memory fit, no checkpointing.
#
#   --m-bh        Fausnaugh et al. 2016 (Paper III, this data's own paper)
#                 adopts 5e7 Msun (Bentz & Katz 2015); Starkey et al. 2017
#                 (Paper VI, "Reverberating Disk Models for NGC 5548", the
#                 CREAM disk-reprocessing fit this package is directly
#                 descended from) adopts 10**7.51 Msun (~3.236e7, from
#                 Pancoast et al. 2014). Using the Paper VI value below,
#                 since it's the directly comparable analysis; check the
#                 current literature before trusting either.
#   (dense NUTS mass matrix: now the default, and worth it for a model
#   this size -- 13 bands' worth of S_band/C_band plus the driver's own
#   Fourier coefficients; see CLAUDE.md decisions #17/#21.)
# ---------------------------------------------------------------------
echo
echo "== Step 2: quick 4-band look (no checkpointing) =="
$RUN_PY scripts/fit_lightcurves.py \
    --m-bh 3.236e7 \
    --band uv1158 1157.5 ngc5548_storm_data/uv1158_band.txt \
    --band u 3472 ngc5548_storm_data/u_band.txt \
    --band g 4776 ngc5548_storm_data/g_band.txt \
    --band z 9157 ngc5548_storm_data/z_band.txt \
    --fit-error-model \
    --num-warmup 400 \
    --num-samples 400 \
    --output-dir ngc5548_quick_look

echo
echo "Report written to ngc5548_quick_look/report.html"

# ---------------------------------------------------------------------
# Step 3: the full 13-band managed/resumable run -- the real analysis.
# Checkpointed, so it can be safely interrupted and resumed (Step 4).
# ---------------------------------------------------------------------
echo
echo "== Step 3: full 13-band managed fit (this is the slow one) =="
$RUN_PY scripts/fit_lightcurves.py \
    --title ngc5548_storm \
    --m-bh 3.236e7 \
    --band uv1158 1157.5 ngc5548_storm_data/uv1158_band.txt \
    --band uv1367 1367.0 ngc5548_storm_data/uv1367_band.txt \
    --band uv1479 1478.5 ngc5548_storm_data/uv1479_band.txt \
    --band uv1746 1746.0 ngc5548_storm_data/uv1746_band.txt \
    --band u 3472 ngc5548_storm_data/u_band.txt \
    --band B 4369 ngc5548_storm_data/B_johnson_band.txt \
    --band g 4776 ngc5548_storm_data/g_band.txt \
    --band V 5404 ngc5548_storm_data/V_johnson_band.txt \
    --band r 6176 ngc5548_storm_data/r_band.txt \
    --band R 6440 ngc5548_storm_data/R_cousins_band.txt \
    --band i 7648 ngc5548_storm_data/i_band.txt \
    --band I 8561 ngc5548_storm_data/I_cousins_band.txt \
    --band z 9157 ngc5548_storm_data/z_band.txt \
    --fit-error-model \
    --num-warmup 1000 \
    --num-samples 3000 \
    --checkpoint-every 200 \
    --report-every 1000 \
    --output-dir outputs

echo
echo "Report written under outputs/ngc5548_storm/ -- open the newest"
echo "run_*/report.html in a browser. Compare the fitted lag-vs-wavelength"
echo "trend against Fausnaugh et al. 2016's own headline result: lags"
echo "follow the lambda^(4/3) disk-reprocessing scaling, but imply a disk"
echo "~3x larger than standard thin-disk theory predicts."

# ---------------------------------------------------------------------
# Step 4: resume it (a no-op here since Step 3 already ran to completion --
# this is what you'd run again after a crash/interruption instead).
# ---------------------------------------------------------------------
echo
echo "== Step 4: resume (no-op, already complete) =="
$RUN_PY scripts/fit_lightcurves.py --title ngc5548_storm --output-dir outputs --resume
