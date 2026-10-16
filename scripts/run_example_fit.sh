#!/usr/bin/env bash
# run_example_fit.sh
# ===================
#
# A worked, runnable example of the whole discotau terminal workflow: make
# some example light curve data, fit it as a managed/resumable run, resume
# it, and (separately) run a quick multi-chain diagnostic fit. Copy/adapt
# the individual `python scripts/fit_lightcurves.py ...` commands below for
# your own light curves -- see `python scripts/fit_lightcurves.py --help`
# for the full argument list, and README.md's "Fitting your own light
# curves" section for what each output file is.
#
# Usage:
#   ./scripts/run_example_fit.sh
#
# Run from the repository root. If you're not using `poetry shell`, prefix
# every `python` call below with `poetry run` instead (already done here).

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_PY="poetry run python"

# ---------------------------------------------------------------------
# Step 1: make some example light curve data (skip this and point at your
# own t/y/yerr text files instead, one per band).
# ---------------------------------------------------------------------
echo "== Step 1: generating example light curve data =="
$RUN_PY scripts/make_example_data.py --outdir example_data
ls -l example_data/

# ---------------------------------------------------------------------
# Step 2: fit it as a managed run -- checkpointed, resumable, everything
# written to outputs/<title>/run_<timestamp>/. This is the mode to use for
# any real, possibly slow/long fitting campaign.
#
#   --title            run name; outputs go to outputs/example_ngc/run_.../
#   --m-bh             fixed black hole mass (solar masses); never inferred
#   --band NAME WAVELENGTH PATH   repeat once per band (lag tied to the
#                       others through log_mdot -- see CLAUDE.md decision #7)
#   --num-warmup / --num-samples  NUTS warmup/sampling iteration counts
#   --checkpoint-every progress is saved to disk every this many samples
#   --report-every     report.html is refreshed every this many new
#                       samples too (optional -- omit to only write it once
#                       at the end)
#   --output-dir        override the output root (default: ./outputs)
# ---------------------------------------------------------------------
echo
echo "== Step 2: managed (checkpointed) fit =="
$RUN_PY scripts/fit_lightcurves.py \
    --title example_ngc \
    --m-bh 1e8 \
    --band g 4770 example_data/g_band.txt \
    --band i 7625 example_data/i_band.txt \
    --num-warmup 300 \
    --num-samples 300 \
    --checkpoint-every 100 \
    --report-every 200 \
    --output-dir outputs

echo
echo "Report written under outputs/example_ngc/ -- open the newest"
echo "run_*/report.html in a browser to see it."

# ---------------------------------------------------------------------
# Step 3: resume it (a no-op here since Step 2 already ran to completion --
# this is what you'd run again after a crash/interruption instead). Only
# --title and --output-dir matter; every other setting (warmup, samples,
# checkpoint-every, ...) is reused from the original run automatically.
# ---------------------------------------------------------------------
echo
echo "== Step 3: resume (no-op, already complete) =="
$RUN_PY scripts/fit_lightcurves.py --title example_ngc --output-dir outputs --resume

# ---------------------------------------------------------------------
# Step 4: a quick, purely in-memory, multi-chain run instead -- useful for
# R-hat/ESS convergence diagnostics rather than a production fit. No
# --title, so nothing is checkpointed, but a one-off report is still
# written to --output-dir.
#
#   --num-chains 4          run 4 independent chains
#   --chain-method vectorized   batch them via vmap on one device, instead
#                            of the (default) one-process-per-chain
#                            "parallel", cheap when the run is dominated by
#                            fixed per-call overhead rather than compute
# ---------------------------------------------------------------------
echo
echo "== Step 4: quick multi-chain diagnostic run (no checkpointing) =="
$RUN_PY scripts/fit_lightcurves.py \
    --m-bh 1e8 \
    --band g 4770 example_data/g_band.txt \
    --band i 7625 example_data/i_band.txt \
    --num-warmup 100 \
    --num-samples 100 \
    --num-chains 4 \
    --chain-method vectorized \
    --output-dir diagnostic_run

echo
echo "Report written to diagnostic_run/report.html"
