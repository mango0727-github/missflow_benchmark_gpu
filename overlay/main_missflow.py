"""MissFlow runner on DiffPuter's benchmark (same datasets / masks / metric).

Drop this file into the DiffPuter repo root and run from there:
    python main_missflow.py --dataname california --split_idx 0

It reuses DiffPuter's own load_dataset / mean_std / get_eval, so the RMSE/MAE are
DIRECTLY comparable to DiffPuter's published numbers, and it additionally reports our
per-cell calibrated coverage (uq_eval) over the MissFlow draws -- the UQ contribution
DiffPuter's accuracy-only evaluation does not measure.

Conventions handled here:
  * DiffPuter mask: 1 = MISSING.   MissFlow wants M = 1 = OBSERVED -> we pass (1 - mask).
  * standardized space X = (x - mean)/std/2 (DiffPuter's), evaluated at *2 (same as theirs).
MissFlow (missflow/ + baselines/) is vendored under ./missflow_pkg and added to sys.path.
"""
import os, sys, time, json, argparse, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # dataset.py + uq_eval.py (the DiffPuter clone)
_pkg = os.environ.get("MISSFLOW_PKG")                      # missflow/ + baselines/ (run_all.sh sets this)
if _pkg:
    sys.path.insert(0, _pkg)

from dataset import load_dataset, mean_std, get_eval        # DiffPuter's code (unchanged)
from uq_eval import evaluate_uq                              # ours
try:
    from baselines.missflow_adapter import MissFlowImputer  # vendored MissFlow
except ImportError as e:
    raise SystemExit("Cannot import MissFlow. Set MISSFLOW_PKG to the dir holding "
                     "missflow/ and baselines/ (the overlay's _vendor).") from e


def main():
    ap = argparse.ArgumentParser(description="MissFlow on the DiffPuter benchmark")
    ap.add_argument("--dataname", default="california")
    ap.add_argument("--split_idx", type=int, default=0)
    ap.add_argument("--mask", default="MCAR", help="MCAR | MAR | MNAR")
    ap.add_argument("--ratio", default="30")
    ap.add_argument("--num_trials", type=int, default=20, help="number of MissFlow draws (m)")
    ap.add_argument("--n_steps", type=int, default=20, help="ODE steps (NFE)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="results_missflow")
    a = ap.parse_args()

    import torch
    device = f"cuda:{a.gpu}" if (a.gpu != -1 and torch.cuda.is_available()) else "cpu"
    mask_type = "MNAR_logistic_T2" if a.mask == "MNAR" else a.mask

    (train_X, test_X, train_mask, test_mask, train_num, test_num,
     train_cat_idx, test_cat_idx, _ext_tr, _ext_te, cat_bin_num) = \
        load_dataset(a.dataname, a.split_idx, mask_type, a.ratio)

    with open(f"datasets/Info/{a.dataname}.json") as f:
        info = json.load(f)
    num_col_idx = info["num_col_idx"]
    num_num = train_num.shape[1]

    mean_X, std_X = mean_std(train_X, train_mask)
    std_X = np.where(std_X == 0, 1.0, std_X)
    Xtr = (train_X - mean_X) / std_X / 2.0
    Xte = (test_X - mean_X) / std_X / 2.0

    # DiffPuter mask is 1=missing; MissFlow expects M=1=observed.
    Mtr_obs = (1 - train_mask).astype(np.float32)
    Mte_obs = (1 - test_mask).astype(np.float32)
    Xobs_tr = np.where(Mtr_obs == 1, Xtr, np.nan).astype(np.float32)
    Xobs_te = np.where(Mte_obs == 1, Xte, np.nan).astype(np.float32)

    imp = MissFlowImputer(n_epochs=a.epochs, n_steps=a.n_steps, device=device, verbose=True)

    t0 = time.time(); imp.fit(Xobs_tr, Mtr_obs); t_train = time.time() - t0

    rows = []
    for tag, X_, M_obs_, Xobs_, full_mask_, cat_idx_ in [
        ("in_sample",     Xtr, Mtr_obs, Xobs_tr, train_mask, train_cat_idx),
        ("out_of_sample", Xte, Mte_obs, Xobs_te, test_mask,  test_cat_idx),
    ]:
        t1 = time.time()
        draws = np.asarray(imp.impute(Xobs_, M_obs_, a.num_trials), dtype=np.float64)  # (T,n,d), std/2
        t_inf = time.time() - t1

        pred_X = draws.mean(0) * 2.0                          # DiffPuter compares in *2 (std) space
        X_true = X_ * 2.0
        mae, rmse = get_eval(a.dataname, pred_X.copy(), X_true, cat_idx_, num_num,
                             cat_bin_num, full_mask_, oos=(tag == "out_of_sample"))

        miss_num = full_mask_[:, num_col_idx].astype(bool)    # 1 = missing (numerical cols)
        uq = evaluate_uq(draws * 2.0, X_true, miss_num, num_num, level=0.95)

        rows.append(dict(dataset=a.dataname, method="MissFlow", mask=mask_type, ratio=a.ratio,
                         split=a.split_idx, sample=tag, m=a.num_trials, nfe=a.n_steps,
                         rmse=float(rmse), mae=float(mae), t_train=round(t_train, 2),
                         t_infer=round(t_inf, 2), t_infer_per_draw=round(t_inf / a.num_trials, 3),
                         **uq))
        print(f"[{tag}] RMSE={rmse:.4f} MAE={mae:.4f} | cov_raw={uq['cov95_raw']:.3f} "
              f"cov_cal={uq['cov95_cal']:.3f} width_cal={uq['width95_cal']:.3f} "
              f"zstd={uq['zstd']:.2f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    out = f"{a.out}/missflow_{a.dataname}_{mask_type}_{a.ratio}_{a.split_idx}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
