"""GPU benchmark driver: run each method through the shared protocol, write one CSV.

Comparable to the paper because it reuses the MissFlow repo's data/masks/metrics
(put the repo on PYTHONPATH; the shell wrapper does this for you).

Examples
  # full GPU head-to-head (MissFlow vs diffusion baselines):
  python run_benchmark.py --device cuda --out results/bench.csv
  # quick local pipeline check (numpy only, no torch/CUDA/datasets needed):
  python run_benchmark.py --datasets synthetic --methods dummy --smoke --out /tmp/t.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np


def get_method_factory(name, device, smoke):
    """Return a zero-arg callable that builds a fresh imputer (fit/impute/name/nfe)."""
    if name == "dummy":                                  # numpy-only: smoke-test the pipeline
        class Dummy:
            name = "dummy"; nfe = 1
            def fit(self, X, M):
                Xc = np.nan_to_num(X, nan=0.0)
                self.mu = Xc.mean(0); self.sd = Xc.std(0) + 1e-3; return self
            def impute(self, X, M, m):
                n, d = X.shape; M = np.asarray(M)
                base = np.where(M == 1, np.nan_to_num(X, nan=0.0), self.mu)
                draws = [base + (M == 0) * np.random.default_rng(s).normal(0, self.sd, (n, d))
                         for s in range(m)]
                return np.stack(draws, 0).astype(np.float32)
        return lambda: Dummy()
    if name == "missflow":
        from baselines.missflow_adapter import MissFlowImputer
        return lambda: MissFlowImputer(n_epochs=(4 if smoke else 400), n_steps=20,
                                       device=device, verbose=True)
    if name == "miwae":
        from baselines.miwae import MIWAEImputer
        return lambda: MIWAEImputer(epochs=(4 if smoke else 200), device=device)
    if name == "gain":
        from baselines.gain import GAINImputer
        return lambda: GAINImputer(epochs=(4 if smoke else 100), device=device)
    if name in ("tabdiff", "missdiff", "diffputer"):
        import bench_diffusion
        return bench_diffusion.build(name, device=device, smoke=smoke)
    raise ValueError(f"unknown method '{name}'")


def load_X(dataset, smoke):
    if dataset == "synthetic":                           # for local pipeline checks
        rng = np.random.default_rng(0)
        A = rng.normal(size=(6, 6))
        n = 200 if smoke else 600
        return rng.multivariate_normal(np.zeros(6), A @ A.T, size=n).astype(np.float64)
    from data.loaders import load_dataset
    return load_dataset(dataset)


def main():
    ap = argparse.ArgumentParser(description="MissFlow GPU benchmark")
    ap.add_argument("--datasets", default="bean,magic,shoppers,letter,california")
    ap.add_argument("--methods", default="missflow,tabdiff,diffputer")
    ap.add_argument("--mechanism", default="mcar", choices=["mcar", "mar", "mnar"])
    ap.add_argument("--p-miss", type=float, default=0.30)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--m", type=int, default=20, help="number of imputations")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="tiny config to verify it runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    datasets = [s for s in a.datasets.split(",") if s]
    methods = [s for s in a.methods.split(",") if s]
    seeds = tuple(int(s) for s in a.seeds.split(","))
    if a.smoke:
        seeds = (0,); a.m = min(a.m, 8)
    out = a.out or f"results/benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    from bench_core import evaluate
    all_rows, failures = [], []
    t_start = time.time()
    for ds in datasets:
        try:
            X = load_X(ds, a.smoke)
        except Exception as e:
            print(f"!! could not load dataset '{ds}': {e}", flush=True); failures.append((ds, "load", str(e))); continue
        for meth in methods:
            print(f"\n=== dataset={ds}  method={meth}  mechanism={a.mechanism} ===", flush=True)
            try:
                fac = get_method_factory(meth, a.device, a.smoke)
                all_rows += evaluate(fac, X, dataset=ds, mechanism=a.mechanism,
                                     p_miss=a.p_miss, seeds=seeds, m=a.m)
            except Exception as e:
                import traceback; traceback.print_exc()
                failures.append((ds, meth, str(e)))
                print(f"!! {ds}/{meth} FAILED: {e}", flush=True)

    if all_rows:
        keys = list(all_rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
        print(f"\n==> wrote {len(all_rows)} rows to {out}  "
              f"({(time.time() - t_start)/60:.1f} min)", flush=True)
    if failures:
        print(f"\n{len(failures)} (dataset,method) combinations failed:", flush=True)
        for ds, meth, e in failures:
            print(f"   {ds}/{meth}: {e}", flush=True)
    sys.exit(0 if all_rows else 1)


if __name__ == "__main__":
    main()
