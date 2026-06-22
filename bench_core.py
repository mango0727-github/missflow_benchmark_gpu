"""Evaluation core (pure numpy): runs ONE method through the paper's protocol and
adds a per-feature split-conformal pass, so every method is scored identically.

Reuses the main MissFlow repo for data/masks/metrics (must be on PYTHONPATH):
    data.masks.make_mask, data.loaders.standardize_train_obs, experiments.metrics
That guarantees the GPU numbers are comparable to the paper's CPU numbers.
"""
from __future__ import annotations
import time
import numpy as np


def per_feature_conformal(comps, Xtrue, miss, rng, level=0.95):
    """Raw percentile band vs per-feature split-conformal calibrated band.

    comps: (m,n,d) standardized completions; miss: (n,d) bool (True=evaluate here).
    Returns (cov_raw, w_raw, cov_cal, w_cal) averaged over missing cells.
    """
    m, n, d = comps.shape
    point = comps.mean(0)
    a = (1.0 - level) / 2.0
    lo = np.percentile(comps, 100 * a, axis=0)
    hi = np.percentile(comps, 100 * (1 - a), axis=0)
    half = (hi - lo) / 2.0 + 1e-8
    inside_raw = (Xtrue >= lo) & (Xtrue <= hi)
    cov_raw = float(inside_raw[miss].mean()) if miss.any() else float("nan")
    w_raw = float((hi - lo)[miss].mean()) if miss.any() else float("nan")

    ins_all, w_all = [], []
    for j in range(d):
        rows = np.where(miss[:, j])[0]
        if len(rows) < 6:                                # too few to calibrate this feature
            continue
        rows = rows.copy(); rng.shuffle(rows)
        k = len(rows) // 2
        cal, ev = rows[:k], rows[k:]
        s = np.abs(Xtrue[cal, j] - point[cal, j]) / half[cal, j]
        qlevel = min(1.0, np.ceil((len(cal) + 1) * level) / len(cal))
        lam = float(np.quantile(s, qlevel, method="higher"))
        loj = point[ev, j] - lam * half[ev, j]
        hij = point[ev, j] + lam * half[ev, j]
        ins_all.append((Xtrue[ev, j] >= loj) & (Xtrue[ev, j] <= hij))
        w_all.append(hij - loj)
    if ins_all:
        cov_cal = float(np.concatenate(ins_all).mean())
        w_cal = float(np.concatenate(w_all).mean())
    else:
        cov_cal, w_cal = cov_raw, w_raw
    return cov_raw, w_raw, cov_cal, w_cal


def evaluate(make_imputer, X, *, dataset, mechanism="mcar", p_miss=0.30,
             seeds=(0, 1, 2), m=20, level=0.95, train_frac=0.70):
    """Run `make_imputer()` over seeds; return a list of per-seed row dicts."""
    from data.masks import make_mask
    from data.loaders import standardize_train_obs
    from experiments.metrics import rmse, mae, zscore_std

    n, d = X.shape
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        M = make_mask(mechanism, X, p_miss, rng)         # 1=observed
        perm = rng.permutation(n)
        n_tr = int(train_frac * n)
        tr, te = perm[:n_tr], perm[n_tr:]
        Xs, _, _ = standardize_train_obs(X, M, tr)
        Xobs = np.where(M == 1, Xs, np.nan).astype(np.float32)

        imp = make_imputer()
        t0 = time.time(); imp.fit(Xobs[tr], M[tr]); t_train = time.time() - t0
        t0 = time.time(); comps = imp.impute(Xobs[te], M[te], m); t_infer = time.time() - t0
        comps = np.asarray(comps, dtype=np.float32)
        point = comps.mean(0)
        miss = (M[te] == 0)

        cov_raw, w_raw, cov_cal, w_cal = per_feature_conformal(
            comps, Xs[te], miss, np.random.default_rng(seed + 9973), level)
        rows.append(dict(
            dataset=dataset, method=getattr(imp, "name", "?"), mechanism=mechanism,
            p_miss=p_miss, seed=seed, n=n, d=d, miss_rate=float((M == 0).mean()),
            rmse=rmse(point, Xs[te], miss), mae=mae(point, Xs[te], miss),
            zstd=zscore_std(comps, Xs[te], miss),
            cov95_raw=cov_raw, width95_raw=w_raw,
            cov95_cal=cov_cal, width95_cal=w_cal,
            nfe=getattr(imp, "nfe", None),
            t_train=round(t_train, 3), t_infer=round(t_infer, 3),
            t_infer_per_draw=round(t_infer / max(m, 1), 4)))
        print(f"  [{dataset}/{rows[-1]['method']}/{mechanism}/seed{seed}] "
              f"rmse={rows[-1]['rmse']:.3f} cov95_cal={cov_cal:.3f} "
              f"width95_cal={w_cal:.3f} nfe={rows[-1]['nfe']} "
              f"t_infer/draw={rows[-1]['t_infer_per_draw']}s", flush=True)
    return rows
