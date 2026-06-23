# MissFlow × DiffPuter benchmark

Compares **MissFlow** against the official **DiffPuter** (ICLR'25) and its bundled
baselines on the *same* datasets / masks / metric, using the **official DiffPuter code**
(so there is no "how did you reproduce the baselines" question), and adds our **per-cell
calibrated coverage** on top of the draws — which DiffPuter's accuracy-only evaluation
does not measure.

## One command (school GPU server)

```bash
git clone https://github.com/mango0727-github/missflow_benchmark_gpu
cd missflow_benchmark_gpu
./run_all.sh
```

`run_all.sh` does everything automatically:
1. finds conda (or installs Miniforge),
2. **clones the official DiffPuter repo** and overlays our files,
3. builds DiffPuter's env (py3.12) and **prepares the datasets/masks** (`download_and_process.py`),
4. builds the MissFlow env (+ torch `CUDA=cu121`) and runs **MissFlow** on those datasets/masks,
5. **aggregates** to `results/comparison.csv` (accuracy + coverage + speed).

Match the server's CUDA: `CUDA=cu118 ./run_all.sh`. Pick a subset:
`DATASETS="magic bean" SPLITS="0 1" ./run_all.sh`.

## Why this is reviewer-proof

- **Accuracy**: MissFlow is scored with DiffPuter's own `get_eval` (standardized-space
  RMSE/MAE on missing cells), so MissFlow's numbers are directly comparable to DiffPuter's
  **published** table — cross-check them.
- **UQ (our contribution)**: `uq_eval.py` computes per-cell 95% coverage + sharpness, raw
  and after per-feature **split conformal**, over the draw stack. For MissFlow it is inline;
  for DiffPuter the optional patch (`overlay/patch_diffputer_uq.py`, applied by `run_all.sh`)
  makes it emit the *same* coverage, so the two are head-to-head.
- **Official baselines**: DiffPuter's repo bundles TabCSDI, MissDiff (`Missdiff_SDE`),
  HyperImpute (MICE/MissForest/GAIN), ReMasker, OT methods. Run them with `RUN_DIFFPUTER=1`
  (or their own commands/envs) and `aggregate.py` folds them in.

## Files

| file | what |
|---|---|
| `run_all.sh` | one-command orchestrator (clone → prep → MissFlow → aggregate) |
| `overlay/main_missflow.py` | MissFlow on DiffPuter's `load_dataset`/`get_eval` (+ our UQ) |
| `overlay/uq_eval.py` | per-cell coverage + split conformal, in DiffPuter's data format |
| `overlay/patch_diffputer_uq.py` | additive patch: make DiffPuter emit the same coverage |
| `aggregate.py` | merge MissFlow + DiffPuter results → `results/comparison.csv` |
| `_vendor/` | vendored MissFlow (`missflow/`, `baselines/`) used by the runner |

## Notes / caveats

- **california** is not auto-downloaded by DiffPuter (no UCI url); drop in
  `DiffPuter/datasets/california/data.csv` to include it. Defaults use the auto-downloaded
  UCI datasets we share (magic, bean, letter, shoppers).
- Running the **DiffPuter** model itself (`RUN_DIFFPUTER=1`) is slow (EM × 10000 epochs ×
  splits). For accuracy you can compare to its *published* numbers; run it only when you
  want its *coverage*.
- DiffPuter needs python 3.12 + its requirements; MissFlow uses our env. `run_all.sh`
  creates and switches both conda envs (the "different environments" the DiffPuter authors note).
- The pre-DiffPuter version of this repo (our own diffusion reimplementations) is retired;
  we now compare against official code.
