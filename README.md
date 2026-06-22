# MissFlow GPU benchmark

Runs the **GPU baselines (diffusion imputers)** head-to-head with **MissFlow** on the
*same* datasets / masks / protocol as the paper, and writes one `.csv` with accuracy,
calibrated coverage, sharpness, NFE, and wall-clock time per draw.

## One click

```bash
git clone https://github.com/mango0727-github/missflow_benchmark_gpu
cd missflow_benchmark_gpu
./run_benchmark.sh
```

**Self-contained**: everything it needs (data loaders, masks, metrics, the MissFlow
model, baselines) is vendored in `_vendor/`, so you only clone this one repo. No second
repo, no `MISSFLOW_REPO` to set.

The script creates `.venv`, installs `requirements.txt`, runs a fast **smoke test**, then
the full run, saving `results/benchmark_<timestamp>.csv`.

Common overrides (env vars):

```bash
SMOKE=1 ./run_benchmark.sh                          # sanity check only
DEVICE=cpu ./run_benchmark.sh                       # no GPU (slow; for testing)
METHODS=missflow,diffputer DATASETS=california ./run_benchmark.sh
MECHANISM=mar SEEDS=0,1,2 M=20 ./run_benchmark.sh
```

## What it runs

| method | what | NFE |
|---|---|---|
| `missflow` | our masked-CFM imputer (reused from the repo, on GPU) | ~20 |
| `tabdiff` / `missdiff` | conditional DDPM for tabular imputation (CSDI / MissDiff family) | 50 |
| `diffputer` | the same denoiser in a diffusion + EM loop (DiffPuter) | 50 |
| `miwae`, `gain` | optional deep baselines, also on GPU | — |
| `dummy` | numpy-only stub, just to smoke-test the pipeline | 1 |

Set the diffusion step count (NFE) in `bench_diffusion.py` (`nfe = 50`); 100–150 gives the
"accurate but slow" regime.

## CSV columns

`dataset, method, mechanism, p_miss, seed, n, d, miss_rate, rmse, mae, zstd,`
`cov95_raw, width95_raw, cov95_cal, width95_cal, nfe, t_train, t_infer, t_infer_per_draw`

- `*_raw` = raw percentile band over draws; `*_cal` = after **per-feature split conformal**
  (so coverage should sit at ~0.95 and `width95_cal` is the apples-to-apples sharpness).
- `t_infer_per_draw` + `nfe` are the speed comparison vs diffusion.

## Notes

- The diffusion baselines are faithful **reference** implementations, not the official
  repos. To use official code, drop in a class with the same `fit(Xobs, M)` /
  `impute(Xobs, M, m) -> (m, n, d)` / `nfe` contract and register it in `bench_diffusion.build`.
- Reproducibility: data/masks/standardization/metrics are the **same code** as the paper
  (vendored in `_vendor/`), so these GPU numbers line up with the paper's CPU numbers.
  `_vendor/` is a snapshot of the main repo; refresh it if that code changes.
