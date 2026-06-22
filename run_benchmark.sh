#!/usr/bin/env bash
# =============================================================================
# MissFlow GPU benchmark — one-click runner for a Linux GPU server.
#
#   1. creates a venv and installs deps (torch + data libs)
#   2. locates the main MissFlow repo (for the shared harness/data/metrics)
#   3. runs a tiny SMOKE test to verify everything is wired
#   4. runs the full benchmark and writes results/benchmark_<timestamp>.csv
#
# Usage:
#   ./run_benchmark.sh                 # full run, defaults (cuda)
#   SMOKE=1 ./run_benchmark.sh         # smoke only (fast sanity check)
#   METHODS=missflow,diffputer DATASETS=california ./run_benchmark.sh
#
# Self-contained: the harness / MissFlow / baselines are vendored in _vendor/,
# so you only need to clone THIS one repo and run.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

# ---- config (override via env) ----------------------------------------------
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-bean,magic,shoppers,letter,california}"
METHODS="${METHODS:-missflow,tabdiff,diffputer}"
MECHANISM="${MECHANISM:-mcar}"
SEEDS="${SEEDS:-0,1,2}"
M="${M:-20}"
SMOKE="${SMOKE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
PYTHON="${PYTHON:-python3}"

echo "==================================================================="
echo " MissFlow GPU benchmark"
echo "   device=$DEVICE  methods=$METHODS  datasets=$DATASETS"
echo "==================================================================="

# ---- 1. python venv + deps --------------------------------------------------
if [ ! -d .venv ]; then
  echo ">> creating venv (.venv)"; "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ "$SKIP_INSTALL" != "1" ]; then
  echo ">> installing requirements (set SKIP_INSTALL=1 to skip)"
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
fi

# ---- 2. self-contained sources (vendored in _vendor/) -----------------------
if [ ! -d "$HERE/_vendor/missflow" ]; then
  echo "!! _vendor/ is missing — this repo is meant to be self-contained."
  echo "   Re-clone the repo (it ships data/, missflow/, baselines/, metrics in _vendor/)."
  exit 2
fi
export PYTHONPATH="$HERE/_vendor:$HERE:${PYTHONPATH:-}"
echo ">> using vendored sources in _vendor/ (no external repo needed)"

# ---- 3. GPU check -----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   GPU: /'
else
  echo "   (no nvidia-smi found; if DEVICE=cuda this will fail — set DEVICE=cpu to test on CPU)"
fi

# ---- 4. smoke test (always; numpy-only pipeline) ----------------------------
echo ">> SMOKE: numpy-only pipeline check (synthetic data, dummy method)"
python run_benchmark.py --datasets synthetic --methods dummy --smoke \
       --device cpu --out results/_smoke_pipeline.csv
echo ">> SMOKE: real methods, tiny config on $DEVICE"
python run_benchmark.py --datasets synthetic --methods "$METHODS" --smoke \
       --device "$DEVICE" --out results/_smoke_methods.csv || {
  echo "!! smoke run of real methods failed — fix before the full run."; exit 3; }
echo ">> smoke OK"

if [ "$SMOKE" = "1" ]; then echo "SMOKE=1 set; stopping after smoke."; exit 0; fi

# ---- 5. full benchmark ------------------------------------------------------
OUT="results/benchmark_$(date +%Y%m%d_%H%M%S).csv"
echo ">> FULL RUN -> $OUT"
python run_benchmark.py --datasets "$DATASETS" --methods "$METHODS" \
       --mechanism "$MECHANISM" --seeds "$SEEDS" --m "$M" \
       --device "$DEVICE" --out "$OUT"
echo "==================================================================="
echo " DONE.  Results: $OUT"
echo "==================================================================="
