#!/usr/bin/env bash
# =============================================================================
# MissFlow GPU benchmark — ZERO-SETUP one-click runner for a Linux GPU server.
#
# Just run this ONE file (double-click it, or `./run_benchmark.sh`). It needs no manual
# setup — it does everything, in order:
#   1. find conda automatically (PATH, HPC `module load`, common install dirs); if there
#      is none, install Miniforge into ~/miniforge3 (one-time). Falls back to a venv.
#   2. create the env + install deps and the matching PyTorch build (CUDA=cu121 default)
#   3. use the vendored sources in _vendor/ (self-contained; no second repo)
#   4. run a tiny SMOKE test, then the full benchmark -> results/benchmark_<ts>.csv
#
# One-time clone on the server, then it is literally one command (or a double-click):
#   git clone https://github.com/mango0727-github/missflow_benchmark_gpu
#   cd missflow_benchmark_gpu && ./run_benchmark.sh
#
# Common overrides (env vars):
#   CUDA=cu118 ./run_benchmark.sh          # match the server's CUDA (cu121/cu118/cu124/cpu)
#   SMOKE=1 ./run_benchmark.sh             # sanity check only (fast)
#   SKIP_INSTALL=1 ./run_benchmark.sh      # reuse an already-built env (faster reruns)
#   USE_CONDA=0 ./run_benchmark.sh         # force a python venv instead of conda
#   NO_BOOTSTRAP=1 ./run_benchmark.sh      # never auto-install conda (use a venv if absent)
#   HOLD=1 ./run_benchmark.sh              # keep the window open at the end (GUI double-click)
#   METHODS=missflow,diffputer DATASETS=california SEEDS=0,1,2 M=20 ./run_benchmark.sh
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
NO_BOOTSTRAP="${NO_BOOTSTRAP:-0}"     # 1 = never auto-install Miniforge
HOLD="${HOLD:-0}"                     # 1 = pause at the end (for GUI double-click)

# CPU device implies CPU torch wheels
if [ "$DEVICE" = "cpu" ]; then CUDA="cpu"; fi

echo "==================================================================="
echo " MissFlow GPU benchmark"
echo "   device=$DEVICE   torch=$CUDA   env=$ENV_NAME"
echo "   methods=$METHODS"
echo "   datasets=$DATASETS"
echo "==================================================================="

# ---- 1. environment: find/install conda (preferred) or use a python venv ----
locate_conda() {                                   # try hard to find an existing conda
  if command -v conda >/dev/null 2>&1; then return 0; fi
  if command -v module >/dev/null 2>&1; then       # HPC environment-modules
    for m in anaconda anaconda3 miniconda miniconda3 miniforge conda; do
      module load "$m" >/dev/null 2>&1 || true
      if command -v conda >/dev/null 2>&1; then return 0; fi
    done
  fi
  for b in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" \
           /opt/conda /opt/miniforge3 /opt/miniconda3 /opt/anaconda3 \
           /usr/local/miniconda3 /usr/local/anaconda3; do
    if [ -x "$b/bin/conda" ]; then export PATH="$b/bin:$PATH"; return 0; fi
  done
  return 1
}
bootstrap_miniforge() {                            # last resort: install conda ourselves
  local dir="$HOME/miniforge3" arch url tmp
  if [ -x "$dir/bin/conda" ]; then export PATH="$dir/bin:$PATH"; return 0; fi
  echo ">> no conda found — installing Miniforge into $dir (one-time, ~100 MB)"
  arch="$(uname -m)"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  tmp="$(mktemp /tmp/miniforge.XXXXXX.sh)"
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$url" -o "$tmp" || return 1
  elif command -v wget >/dev/null 2>&1; then wget -qO "$tmp" "$url" || return 1
  else echo "!! need curl or wget to auto-install conda"; return 1; fi
  bash "$tmp" -b -p "$dir" >/dev/null 2>&1 || { echo "!! Miniforge install failed"; return 1; }
  rm -f "$tmp"; export PATH="$dir/bin:$PATH"
  command -v conda >/dev/null 2>&1
}

use_conda=0
if [ "$USE_CONDA" != "0" ]; then
  if locate_conda; then use_conda=1
  elif [ "$NO_BOOTSTRAP" != "1" ] && bootstrap_miniforge; then use_conda=1
  elif [ "$USE_CONDA" = "1" ]; then
    echo "!! USE_CONDA=1 but conda could not be found or installed."; exit 2
  else
    echo ">> conda unavailable; falling back to a python venv"
  fi
fi

if [ "$use_conda" = "1" ]; then
  echo ">> conda: $(command -v conda)   env: $ENV_NAME"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">> creating conda env '$ENV_NAME'"
    if [ -f environment.yml ]; then conda env create -n "$ENV_NAME" -f environment.yml
    else conda create -y -n "$ENV_NAME" python=3.11 pip; fi
  fi
  conda activate "$ENV_NAME"
else
  echo ">> python venv: .venv"
  command -v "$PYTHON" >/dev/null 2>&1 || { echo "!! no '$PYTHON' and no conda — cannot build an env."; exit 2; }
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
if [ "$HOLD" = "1" ]; then read -rp "Press Enter to close..." _; fi
