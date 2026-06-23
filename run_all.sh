#!/usr/bin/env bash
# =============================================================================
# MissFlow x DiffPuter benchmark — official-code comparison, one entry point.
#
# Clones the official DiffPuter repo (ICLR'25), overlays MissFlow + our per-cell
# coverage eval, runs everything on the SAME datasets / masks / metric, aggregates
# to one CSV. Two conda envs are used (DiffPuter needs python 3.12; MissFlow uses ours),
# created and switched automatically.
#
#   ./run_all.sh                      # clone, prep data, run MissFlow, aggregate
#   RUN_DIFFPUTER=1 ./run_all.sh      # ALSO run the DiffPuter baseline (long; for UQ)
#   DATASETS="magic bean" SPLITS="0 1" ./run_all.sh
#   CUDA=cu118 ./run_all.sh           # match the server's CUDA
#
# Note: california is not auto-downloaded by DiffPuter (no UCI url); provide its data.csv
# manually to include it. Defaults use the auto-downloaded UCI datasets we share.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"; REPO="$(pwd)"

DIFFPUTER_URL="${DIFFPUTER_URL:-https://github.com/hengruizhang98/DiffPuter}"
DP="${DP:-$REPO/DiffPuter}"
DATASETS="${DATASETS:-magic bean letter shoppers}"
SPLITS="${SPLITS:-0 1 2}"
MASK="${MASK:-MCAR}"
M="${M:-20}"; NFE="${NFE:-20}"; EPOCHS="${EPOCHS:-400}"
CUDA="${CUDA:-cu121}"
ENV_NAME="${ENV_NAME:-missflow_bench}"      # our env (MissFlow)
DP_ENV="${DP_ENV:-diffputer}"               # DiffPuter's env (python 3.12)
RUN_DIFFPUTER="${RUN_DIFFPUTER:-0}"
APPLY_UQ_PATCH="${APPLY_UQ_PATCH:-1}"

# ---- conda: find it, or install Miniforge (same as before) ------------------
locate_conda() {
  command -v conda >/dev/null 2>&1 && return 0
  if command -v module >/dev/null 2>&1; then
    for m in anaconda anaconda3 miniconda miniconda3 miniforge conda; do
      module load "$m" >/dev/null 2>&1 || true
      command -v conda >/dev/null 2>&1 && return 0
    done
  fi
  for b in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -x "$b/bin/conda" ] && { export PATH="$b/bin:$PATH"; return 0; }
  done
  return 1
}
locate_conda || { echo "!! conda not found; load it (module load anaconda) and retry"; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
ensure_env() { conda env list | awk '{print $1}' | grep -qx "$1" || conda create -y -n "$1" python="$2" pip; }

# ---- 1. clone DiffPuter + overlay our files ---------------------------------
[ -d "$DP/.git" ] || git clone "$DIFFPUTER_URL" "$DP"
cp "$REPO/overlay/uq_eval.py" "$REPO/overlay/main_missflow.py" "$DP/"
if [ "$APPLY_UQ_PATCH" = "1" ]; then
  python "$REPO/overlay/patch_diffputer_uq.py" "$DP/main.py" || true
fi

# ---- 2. DiffPuter env + prepare datasets/masks (their code, their env) -------
echo ">> [DiffPuter env] preparing datasets/masks"
ensure_env "$DP_ENV" 3.12
conda activate "$DP_ENV"
pip install -q -r "$DP/requirements/diffputer.txt" || pip install -q pandas numpy scikit-learn scipy openpyxl xlrd tqdm requests
( cd "$DP" && [ -d datasets/magic/masks ] || python download_and_process.py )
conda deactivate

# ---- 3. MissFlow env + run MissFlow on the prepared data --------------------
echo ">> [MissFlow env] running MissFlow"
ensure_env "$ENV_NAME" 3.11
conda activate "$ENV_NAME"
pip install -q numpy pandas scikit-learn scipy
python -c "import torch" 2>/dev/null || pip install -q torch --index-url "https://download.pytorch.org/whl/${CUDA}"
export MISSFLOW_PKG="$REPO/_vendor"
mkdir -p "$REPO/results/missflow"
for ds in $DATASETS; do for s in $SPLITS; do
  echo ">> MissFlow  dataset=$ds  split=$s"
  ( cd "$DP" && python main_missflow.py --dataname "$ds" --split_idx "$s" --mask "$MASK" \
       --num_trials "$M" --n_steps "$NFE" --epochs "$EPOCHS" --out "$REPO/results/missflow" )
done; done

# ---- 4. (optional) run the DiffPuter baseline (long; gives its coverage) -----
if [ "$RUN_DIFFPUTER" = "1" ]; then
  echo ">> [DiffPuter env] running DiffPuter baseline (this is slow)"
  conda deactivate; conda activate "$DP_ENV"
  for ds in $DATASETS; do for s in $SPLITS; do
    ( cd "$DP" && python main.py --dataname "$ds" --split_idx "$s" --mask "$MASK" \
         --num_trials "$M" --num_steps 50 )
  done; done
  conda deactivate; conda activate "$ENV_NAME"
fi

# ---- 5. aggregate -----------------------------------------------------------
python "$REPO/aggregate.py" --missflow "$REPO/results/missflow" \
       --diffputer "$DP/results" --out "$REPO/results/comparison.csv"
echo "==================================================================="
echo " DONE.  Comparison: $REPO/results/comparison.csv"
echo "   (cross-check the MissFlow RMSE against DiffPuter's PUBLISHED table.)"
echo "==================================================================="
