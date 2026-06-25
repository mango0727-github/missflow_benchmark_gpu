#!/usr/bin/env bash
# =============================================================================
# MissFlow vs the diffusion imputers (DiffPuter, MissDiff, TabCSDI) — official code.
#
# Clones the official DiffPuter repo (ICLR'25, which bundles MissDiff + TabCSDI), runs
# MissFlow + ALL THREE diffusion baselines on the SAME prepared datasets/masks/metric,
# and aggregates accuracy (+ coverage where available) into one CSV. This closes the
# accuracy/NFE/wall-clock comparison the paper names as the remaining gap.
#
#   ./run_all.sh                          # full: prep data, MissFlow + 3 diffusion baselines
#   RUN_DIFFUSION=0 ./run_all.sh          # MissFlow only
#   DATASETS="magic bean" SPLITS="0 1" BASELINE_EPOCHS=200 ./run_all.sh   # quick smoke
#   CUDA=cu118 ./run_all.sh               # match the server's CUDA
#   AUTO_STOP="runpodctl stop pod $RUNPOD_POD_ID" ./run_all.sh   # self-stop when done (rented cloud)
#
# Three conda envs are created/switched automatically (the DiffPuter authors split them):
#   $ENV_NAME  : MissFlow (py3.11 + our torch)
#   $DP_ENV    : DiffPuter + MissDiff (py3.12, modern torch)
#   $TABCSDI_ENV: TabCSDI (py3.10, torch 1.13 / numpy 1.23 — its pinned old stack)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"; REPO="$(pwd)"

DIFFPUTER_URL="${DIFFPUTER_URL:-https://github.com/hengruizhang98/DiffPuter}"
DP="${DP:-$REPO/DiffPuter}"
DATASETS="${DATASETS:-magic bean letter shoppers}"
SPLITS="${SPLITS:-0 1 2}"
MASK="${MASK:-MAR}"          # MAR: DiffPuter's published Tables 7/8 cover it -> self-validating
M="${M:-20}"; NFE="${NFE:-20}"; EPOCHS="${EPOCHS:-400}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-0}"          # >0 shrinks baseline epochs (smoke only)
CUDA="${CUDA:-cu121}"; TABCSDI_CUDA="${TABCSDI_CUDA:-cu117}"
ENV_NAME="${ENV_NAME:-missflow_bench}"
DP_ENV="${DP_ENV:-diffputer}"
TABCSDI_ENV="${TABCSDI_ENV:-tabcsdi}"
RUN_DIFFUSION="${RUN_DIFFUSION:-1}"
APPLY_UQ_PATCH="${APPLY_UQ_PATCH:-1}"
NSPLITS=$(echo $SPLITS | wc -w)

# ---- conda: find it (or install Miniforge) ----------------------------------
locate_conda() {
  command -v conda >/dev/null 2>&1 && return 0
  if command -v module >/dev/null 2>&1; then
    for m in anaconda anaconda3 miniconda miniconda3 miniforge conda; do
      module load "$m" >/dev/null 2>&1 || true; command -v conda >/dev/null 2>&1 && return 0
    done
  fi
  for b in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -x "$b/bin/conda" ] && { export PATH="$b/bin:$PATH"; return 0; }
  done
  return 1
}
bootstrap_miniforge() {                          # fresh box (e.g. RunPod): install conda
  local dir="$HOME/miniforge3" arch url tmp
  [ -x "$dir/bin/conda" ] && { export PATH="$dir/bin:$PATH"; return 0; }
  echo ">> no conda found -- installing Miniforge into $dir (one-time, ~1-2 min)"
  arch="$(uname -m)"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  tmp="$(mktemp /tmp/miniforge.XXXXXX.sh)"
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$url" -o "$tmp" || return 1
  elif command -v wget >/dev/null 2>&1; then wget -qO "$tmp" "$url" || return 1
  else echo "!! need curl or wget to install conda"; return 1; fi
  bash "$tmp" -b -p "$dir" >/dev/null 2>&1 || { echo "!! Miniforge install failed"; return 1; }
  rm -f "$tmp"; export PATH="$dir/bin:$PATH"; command -v conda >/dev/null 2>&1
}
locate_conda || bootstrap_miniforge || { echo "!! conda not found and could not install it"; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
ensure_env() { conda env list | awk '{print $1}' | grep -qx "$1" || conda create -y -n "$1" python="$2" pip; }
subset() {  # $1=file ; trims its internal dataset/mask/split loops to our subset
  local ep=""; [ "$BASELINE_EPOCHS" != "0" ] && ep="--epochs $BASELINE_EPOCHS"
  python "$REPO/overlay/set_subset.py" "$1" --datasets "$DATASETS" --masks "$MASK" --splits "$NSPLITS" $ep
}

# ---- 1. clone DiffPuter + overlay our files ---------------------------------
[ -d "$DP/.git" ] || git clone "$DIFFPUTER_URL" "$DP"
cp "$REPO/overlay/uq_eval.py" "$REPO/overlay/main_missflow.py" "$DP/"
[ "$APPLY_UQ_PATCH" = "1" ] && python "$REPO/overlay/patch_diffputer_uq.py" "$DP/main.py" || true

# ---- 2. DiffPuter env + prepare the shared datasets/masks (their code) -------
echo ">> [diffputer env] preparing shared datasets/masks"
ensure_env "$DP_ENV" 3.12; conda activate "$DP_ENV"
pip install -q -r "$DP/requirements/diffputer.txt" 2>/dev/null || \
  pip install -q pandas numpy scikit-learn scipy openpyxl xlrd tqdm requests pyyaml torch
( cd "$DP" && [ -d datasets/magic/masks ] || python download_and_process.py )
conda deactivate

# ---- 3. MissFlow (our env) on the prepared data -----------------------------
echo ">> [missflow env] running MissFlow"
ensure_env "$ENV_NAME" 3.11; conda activate "$ENV_NAME"
pip install -q numpy pandas scikit-learn scipy
python -c "import torch" 2>/dev/null || pip install -q torch --index-url "https://download.pytorch.org/whl/${CUDA}"
export MISSFLOW_PKG="$REPO/_vendor"; mkdir -p "$REPO/results/missflow"
for ds in $DATASETS; do for s in $SPLITS; do
  echo ">> MissFlow  $ds  split $s"
  ( cd "$DP" && python main_missflow.py --dataname "$ds" --split_idx "$s" --mask "$MASK" \
       --num_trials "$M" --n_steps "$NFE" --epochs "$EPOCHS" --out "$REPO/results/missflow" )
done; done
conda deactivate

# ---- 4. diffusion baselines (official code) ---------------------------------
if [ "$RUN_DIFFUSION" = "1" ]; then
  echo ">> [diffputer env] DiffPuter + MissDiff"
  conda activate "$DP_ENV"
  for ds in $DATASETS; do for s in $SPLITS; do
    echo ">> DiffPuter $ds split $s"
    ( cd "$DP" && python main.py --dataname "$ds" --split_idx "$s" --mask "$MASK" )
  done; done
  subset "$DP/baselines/Missdiff_SDE/Missdiff_benchmark.py"
  echo ">> MissDiff (runs the subset internally)"
  ( cd "$DP/baselines/Missdiff_SDE" && python Missdiff_benchmark.py ) || echo "!! MissDiff failed (check its env)"
  conda deactivate

  echo ">> [tabcsdi env] TabCSDI (old torch stack)"
  ensure_env "$TABCSDI_ENV" 3.10; conda activate "$TABCSDI_ENV"
  pip install -q -r "$DP/baselines/TabCSDI/requirements.txt" 2>/dev/null || true
  python -c "import torch" 2>/dev/null || \
    pip install -q torch==1.13.0 --index-url "https://download.pytorch.org/whl/${TABCSDI_CUDA}" || \
    pip install -q torch==1.13.0
  subset "$DP/baselines/TabCSDI/csdi_benchmark.py"
  echo ">> TabCSDI (runs the subset internally)"
  ( cd "$DP/baselines/TabCSDI" && python csdi_benchmark.py --config uci.yaml --nsample "$M" ) \
    || echo "!! TabCSDI failed (check its old-torch env / CUDA tag TABCSDI_CUDA)"
  conda deactivate
  conda activate "$ENV_NAME"
fi

# ---- 5. aggregate -----------------------------------------------------------
python "$REPO/aggregate.py" --missflow "$REPO/results/missflow" \
       --diffputer-root "$DP" --out "$REPO/results/comparison.csv"

# ---- 6. reproduction gate: do the baselines match the published tables? ------
if [ -f "$REPO/published_reference.csv" ]; then
  echo ">> reproduction gate vs DiffPuter ICLR'25 published Tables 7/8 ($MASK)"
  python "$REPO/check_repro.py" --reproduced "$REPO/results/comparison.csv" \
         --reference "$REPO/published_reference.csv" --tol "${REPRO_TOL:-0.10}" || true
fi
echo "==================================================================="
echo " DONE.  Comparison: $REPO/results/comparison.csv"
echo "   PASS rows above = our harness reproduces the paper -> the setup (and MissFlow's"
echo "   numbers on it) are trustworthy.  Run with MASK=MAR for the published-table gate."
echo "==================================================================="

# ---- self-stop when finished (avoid idle billing on a rented cloud GPU) ------
# Set AUTO_STOP to the command that stops THIS instance, e.g. on RunPod:
#   AUTO_STOP="runpodctl stop pod $RUNPOD_POD_ID" ./run_all.sh
if [ -n "${AUTO_STOP:-}" ]; then
  echo ">> finished; AUTO_STOP -> $AUTO_STOP"
  sleep 5
  eval "$AUTO_STOP" || echo "!! AUTO_STOP command failed; stop the pod manually to avoid charges."
fi
