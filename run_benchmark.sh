#!/usr/bin/env bash
# =============================================================================
# MissFlow GPU benchmark — one-click runner for a Linux GPU server.
#
#   1. creates an isolated env (CONDA if available, else a python venv) + installs deps
#   2. installs the matching PyTorch build (CUDA=cu121 by default; CUDA=cpu to test)
#   3. uses the vendored sources in _vendor/ (self-contained; no second repo)
#   4. runs a tiny SMOKE test, then the full benchmark -> results/benchmark_<ts>.csv
#
# One click on the school GPU server (conda is auto-detected):
#   git clone https://github.com/mango0727-github/missflow_benchmark_gpu
#   cd missflow_benchmark_gpu
#   ./run_benchmark.sh
#
# Common overrides (env vars):
#   CUDA=cu118 ./run_benchmark.sh          # match the server's CUDA (cu121/cu118/cu124/cpu)
#   ENV_NAME=missflow_bench ./run_benchmark.sh
#   USE_CONDA=0 ./run_benchmark.sh         # force a python venv instead of conda
#   SMOKE=1 ./run_benchmark.sh             # sanity check only (fast)
#   SKIP_INSTALL=1 ./run_benchmark.sh      # reuse an already-built env (faster reruns)
#   METHODS=missflow,diffputer DATASETS=california ./run_benchmark.sh
#   MECHANISM=mar SEEDS=0,1,2 M=20 ./run_benchmark.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

# ---- config (override via env) ----------------------------------------------
DEVICE="${DEVICE:-cuda}"
CUDA="${CUDA:-cu121}"                 # torch wheel tag: cu121 / cu118 / cu124 / cpu
ENV_NAME="${ENV_NAME:-missflow_bench}"
USE_CONDA="${USE_CONDA:-auto}"        # auto | 1 | 0
DATASETS="${DATASETS:-bean,magic,shoppers,letter,california}"
METHODS="${METHODS:-missflow,tabdiff,diffputer}"
MECHANISM="${MECHANISM:-mcar}"
SEEDS="${SEEDS:-0,1,2}"
M="${M:-20}"
SMOKE="${SMOKE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
PYTHON="${PYTHON:-python3}"

# CPU device implies CPU torch wheels
if [ "$DEVICE" = "cpu" ]; then CUDA="cpu"; fi

echo "==================================================================="
echo " MissFlow GPU benchmark"
echo "   device=$DEVICE   torch=$CUDA   env=$ENV_NAME"
echo "   methods=$METHODS"
echo "   datasets=$DATASETS"
echo "==================================================================="

# ---- 1. environment: conda (preferred) or python venv -----------------------
have_conda=0
command -v conda >/dev/null 2>&1 && have_conda=1
case "$USE_CONDA" in
  1) use_conda=1 ;;
  0) use_conda=0 ;;
  *) use_conda=$have_conda ;;
esac

if [ "$use_conda" = "1" ]; then
  if [ "$have_conda" != "1" ]; then
    echo "!! USE_CONDA=1 but 'conda' is not on PATH. Load it first (e.g. 'module load anaconda')."; exit 2
  fi
  echo ">> conda env: $ENV_NAME"
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">> creating conda env '$ENV_NAME'"
    if [ -f environment.yml ]; then
      conda env create -n "$ENV_NAME" -f environment.yml
    else
      conda create -y -n "$ENV_NAME" python=3.11 pip
    fi
  fi
  conda activate "$ENV_NAME"
else
  echo ">> python venv: .venv"
  [ -d .venv ] || "$PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ---- 2. deps: pure-python + the matching torch build ------------------------
if [ "$SKIP_INSTALL" != "1" ]; then
  echo ">> installing deps (set SKIP_INSTALL=1 to skip on reruns)"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  echo ">> installing torch ($CUDA) from the official wheel index"
  python -m pip install --quiet torch --index-url "https://download.pytorch.org/whl/${CUDA}" || {
    echo "!! torch install failed for CUDA=$CUDA. Pick the tag matching the server's driver:"
    echo "   CUDA=cu118 / cu121 / cu124  (or CUDA=cpu to test). Check 'nvidia-smi'."; exit 4; }
fi

# ---- 3. self-contained sources (vendored in _vendor/) -----------------------
if [ ! -d "$HERE/_vendor/missflow" ]; then
  echo "!! _vendor/ is missing — re-clone (this repo ships data/, missflow/, baselines/ in _vendor/)."; exit 2
fi
export PYTHONPATH="$HERE/_vendor:$HERE:${PYTHONPATH:-}"

# ---- 4. environment report (fail fast on a CUDA mismatch) -------------------
DEVICE="$DEVICE" CUDA="$CUDA" python - <<'PY'
import os, sys, torch
dev, cuda = os.environ.get("DEVICE", "cuda"), os.environ.get("CUDA", "cu121")
ok = torch.cuda.is_available()
msg = f"   torch {torch.__version__}   cuda_available={ok}"
if ok:
    msg += f"   gpu={torch.cuda.get_device_name(0)}"
print(msg)
if dev == "cuda" and not ok:
    print("!! DEVICE=cuda but torch cannot see a GPU. Fix one of:")
    print(f"   - run on a GPU node (this looks like a CPU node), or")
    print(f"   - the torch build (CUDA={cuda}) does not match the driver: set CUDA to the")
    print( "     tag matching 'nvidia-smi' (cu118/cu121/cu124) and rerun WITHOUT SKIP_INSTALL, or")
    print( "   - just test the pipeline on CPU with DEVICE=cpu.")
    sys.exit(7)
PY
if [ "$DEVICE" = "cuda" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   GPU: /'
  else
    echo "   (no nvidia-smi found; if this is a CPU node set DEVICE=cpu)"
  fi
fi

# ---- 5. smoke test (always; catches wiring problems before the long run) ----
echo ">> SMOKE: numpy-only pipeline check"
python run_benchmark.py --datasets synthetic --methods dummy --smoke \
       --device cpu --out results/_smoke_pipeline.csv
echo ">> SMOKE: real methods, tiny config on $DEVICE"
python run_benchmark.py --datasets synthetic --methods "$METHODS" --smoke \
       --device "$DEVICE" --out results/_smoke_methods.csv || {
  echo "!! smoke run of real methods failed — fix before the full run."; exit 3; }
echo ">> smoke OK"
if [ "$SMOKE" = "1" ]; then echo "SMOKE=1 set; stopping after smoke."; exit 0; fi

# ---- 6. full benchmark ------------------------------------------------------
OUT="results/benchmark_$(date +%Y%m%d_%H%M%S).csv"
echo ">> FULL RUN -> $OUT"
python run_benchmark.py --datasets "$DATASETS" --methods "$METHODS" \
       --mechanism "$MECHANISM" --seeds "$SEEDS" --m "$M" \
       --device "$DEVICE" --out "$OUT"
echo "==================================================================="
echo " DONE.  Results: $OUT"
echo "==================================================================="
