"""Per-cell UQ evaluation for imputation draws, in DiffPuter's data format.

DiffPuter (and the MissFlow runner that mirrors it) produce `num_trials` imputed
matrices per dataset/mask. DiffPuter then *averages* them and reports only RMSE/MAE.
Our contribution is the calibrated predictive interval, so this module operates on the
full stack of draws (before averaging) and reports coverage + sharpness, raw and after
per-feature split conformal.

Conventions (match DiffPuter exactly):
  * layout  : columns are [ numerical (num_num) | binary-encoded categorical ].
              We evaluate UQ on the NUMERICAL columns only (continuous intervals).
  * mask    : mask == 1 means MISSING (DiffPuter convention).
  * space   : everything is in DiffPuter's standardized space (X = (x-mean)/std/2 then
              compared at *2), so widths are comparable across methods.

Metrics over the missing numerical cells:
  zstd        : std of z = (true - mean)/draw_std   (>1 => draws too tight)
  cov95_raw   : coverage of the [alpha/2, 1-alpha/2] percentile band over draws
  width95_raw : mean width of that band
  cov95_cal   : coverage after per-feature split conformal
  width95_cal : mean width after conformal
"""
from __future__ import annotations
import numpy as np


def _bands(draws, level):
    a = (1.0 - level) / 2.0
    lo = np.percentile(draws, 100 * a, axis=0)
    hi = np.percentile(draws, 100 * (1 - a), axis=0)
    return lo, hi, draws.mean(0), draws.std(0) + 1e-9


def evaluate_uq(draws, X_true, miss_mask_num, num_num, level=0.95, seed=0):
    """Compute per-cell UQ metrics on the numerical missing cells.

    Args:
      draws        : (T, n, d) stack of imputed matrices (standardized space).
      X_true       : (n, d) ground-truth matrix (same standardized space).
      miss_mask_num: (n, num_num) with 1 = missing (numerical columns only).
      num_num      : number of leading numerical columns in d.
    Returns: dict of metrics (floats) + n_missing.
    """
    rng = np.random.default_rng(seed)
    D = np.asarray(draws)[:, :, :num_num].astype(np.float64)   # (T, n, num_num)
    Xt = np.asarray(X_true)[:, :num_num].astype(np.float64)    # (n, num_num)
    miss = np.asarray(miss_mask_num).astype(bool)              # (n, num_num)

    lo, hi, mu, sd = _bands(D, level)
    half = (hi - lo) / 2.0 + 1e-9
    inside_raw = (Xt >= lo) & (Xt <= hi)

    if miss.sum() == 0:
        return dict(zstd=float("nan"), cov95_raw=float("nan"), width95_raw=float("nan"),
                    cov95_cal=float("nan"), width95_cal=float("nan"), n_missing=0)

    cov_raw = float(inside_raw[miss].mean())
    width_raw = float((hi - lo)[miss].mean())
    z = (Xt - mu) / sd
    zstd = float(z[miss].std())

    # per-feature split conformal: calibrate lambda on half the missing cells of a
    # feature, evaluate coverage on the other half (marginal finite-sample guarantee).
    ins, wid = [], []
    for j in range(num_num):
        r = np.where(miss[:, j])[0]
        if len(r) < 6:                                    # too few to calibrate -> raw
            ins.append(inside_raw[r, j]); wid.append(hi[r, j] - lo[r, j]); continue
        r = r.copy(); rng.shuffle(r); k = len(r) // 2
        cal, ev = r[:k], r[k:]
        s = np.abs(Xt[cal, j] - mu[cal, j]) / half[cal, j]
        q = min(1.0, np.ceil((len(cal) + 1) * level) / len(cal))
        lam = np.quantile(s, q, method="higher")
        ins.append(np.abs(Xt[ev, j] - mu[ev, j]) <= lam * half[ev, j])
        wid.append(2 * lam * half[ev, j])
    cov_cal = float(np.concatenate(ins).mean())
    width_cal = float(np.concatenate(wid).mean())

    return dict(zstd=zstd, cov95_raw=cov_raw, width95_raw=width_raw,
                cov95_cal=cov_cal, width95_cal=width_cal, n_missing=int(miss.sum()))


# helper: pull the numerical missing mask out of DiffPuter's full mask --------------
def numerical_miss_mask(full_mask, num_col_idx):
    """full_mask: (n, n_original_cols) with 1=missing; returns (n, len(num_col_idx))."""
    return np.asarray(full_mask)[:, num_col_idx].astype(bool)
