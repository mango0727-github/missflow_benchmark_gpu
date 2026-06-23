# MissFlow vs the diffusion imputers (official code)

Runs **MissFlow** head-to-head with **all three diffusion imputers the paper names as the
remaining gap** — **DiffPuter** (ICLR'25), **MissDiff**, **TabCSDI** — using their
**official code**, on the **same** prepared datasets / masks / metric, and aggregates
accuracy (+ NFE + coverage where available) into one CSV. DiffPuter's repo bundles MissDiff
(`Missdiff_SDE`) and TabCSDI, and all three read the *same* root `datasets/`, so the
comparison is apples-to-apples by construction.

## One command (school GPU server)

```bash
git clone https://github.com/mango0727-github/missflow_benchmark_gpu
cd missflow_benchmark_gpu
./run_all.sh
```

`run_all.sh` automatically: finds conda → clones official DiffPuter → prepares the shared
datasets/masks → runs **MissFlow + DiffPuter + MissDiff + TabCSDI** → writes
`results/comparison.csv` (accuracy + NFE + coverage). Useful overrides:

```bash
RUN_DIFFUSION=0 ./run_all.sh                                   # MissFlow only
DATASETS="magic bean" SPLITS="0 1" BASELINE_EPOCHS=200 ./run_all.sh   # quick smoke
CUDA=cu118 TABCSDI_CUDA=cu117 ./run_all.sh                     # match the server's CUDA
```

## Three conda envs (created/switched automatically)

The DiffPuter authors split the baselines across environments; `run_all.sh` builds them:

| env | methods | stack |
|---|---|---|
| `missflow_bench` | MissFlow | py3.11 + our torch (`CUDA`) |
| `diffputer` | DiffPuter, MissDiff | py3.12, modern torch |
| `tabcsdi` | TabCSDI | py3.10, **torch 1.13 / numpy 1.23** (its pinned old stack, `TABCSDI_CUDA`) |

## Why this is reviewer-proof

- **Accuracy**: every method is scored with DiffPuter's own `get_eval` (standardized-space
  RMSE/MAE on missing cells), so the numbers are directly comparable to DiffPuter's
  **published** table — cross-check them. NFE per method (MissFlow 20, DiffPuter 50,
  MissDiff 50, TabCSDI 100) gives the accuracy/NFE/wall-clock Pareto.
- **UQ (our contribution)**: `uq_eval.py` adds per-cell 95% coverage + sharpness, raw and
  after per-feature split conformal, over the draw stack. Inline for MissFlow; for DiffPuter
  via the optional patch. (MissDiff/TabCSDI coverage is a follow-up; their accuracy/NFE are
  the paper's stated gap.)
- **Official baselines, same data**: all baselines are the authors' code reading the shared
  `datasets/`; we only trim their internal dataset/mask/split loops to our subset
  (`overlay/set_subset.py`).

## Reproduction gate (this must pass first)

Before the comparison means anything, the baselines must **reproduce their published
numbers** on this harness — that is what proves the setup is correct (and that MissFlow's
numbers on the *same* harness are credible). `run_all.sh` runs this automatically at the end:

```
PASS DiffPuter california MAR in rmse  pub=0.571 repro=0.577  1.1%
PASS TabCSDI   california MAR in rmse  pub=1.116 repro=1.100  1.5%
FAIL MissDiff  bean       MAR in rmse  pub=0.973 repro=1.500 54.2%   <- fix before trusting
```

- The reference (`published_reference.csv`) holds DiffPuter ICLR'25 **Table 7/8** numbers
  (MAR, in-sample, RMSE+MAE) for DiffPuter / TabCSDI / MissDiff on our datasets. That is why
  the default mask is **MAR** — it is the protocol with a full published numerical table.
- `check_repro.py` averages our reproduced numbers over splits and flags PASS/FAIL by
  relative tolerance (`REPRO_TOL`, default 10%). FAIL = our protocol differs from the paper
  there; fix it before trusting any comparison.

## Files

| file | what |
|---|---|
| `run_all.sh` | one command: clone → prep → MissFlow + 3 diffusion baselines → aggregate |
| `overlay/main_missflow.py` | MissFlow on DiffPuter's `load_dataset`/`get_eval` (+ our UQ) |
| `overlay/uq_eval.py` | per-cell coverage + split conformal, in DiffPuter's data format |
| `overlay/set_subset.py` | trim a baseline runner's internal loops to our datasets/masks/splits |
| `overlay/patch_diffputer_uq.py` | additive patch: make DiffPuter emit the same coverage |
| `aggregate.py` | merge MissFlow + DiffPuter + MissDiff + TabCSDI → `results/comparison.csv` |
| `check_repro.py` | reproduction gate: harness numbers vs the published table (PASS/FAIL) |
| `published_reference.csv` | DiffPuter ICLR'25 Table 7/8 numbers (MAR in-sample RMSE/MAE) |
| `_vendor/` | vendored MissFlow (`missflow/`, `baselines/`) used by the runner |

## Notes / caveats

- The diffusion baselines run their **full** training (≈10000 epochs × EM × splits) to match
  their published numbers — this is **slow on GPU**. Use `BASELINE_EPOCHS` + a small
  `DATASETS`/`SPLITS` for a smoke first.
- **TabCSDI** needs an old torch (1.13) / numpy (1.23) stack; its CUDA wheel tag is
  `TABCSDI_CUDA` (default `cu117`). If the server's driver is too new for cu117, it may need
  a CPU fallback or a container.
- **california** is not auto-downloaded by DiffPuter (no UCI url); drop in its `data.csv` to
  include it. Defaults use the auto-downloaded UCI datasets we share (magic, bean, letter, shoppers).
- The pre-DiffPuter version of this repo (our own reimplementations) is retired.
