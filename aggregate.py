"""Merge MissFlow results (our CSVs) and DiffPuter/baseline results (their result.txt,
plus optional UQ json from the patch) into ONE comparison CSV.

  --missflow DIR : folder of missflow_*.csv written by main_missflow.py
  --diffputer DIR: DiffPuter's results/ tree (result.txt per dataset/mask/split)
  --out FILE     : combined comparison CSV

Accuracy is the same metric for both (DiffPuter's get_eval, standardized-space RMSE/MAE),
so the rows are directly comparable. Coverage columns are filled for methods that emit a
draw stack (MissFlow always; DiffPuter/baselines only if the UQ patch saved uq_*.json).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re

FIELDS = ["dataset", "method", "mask", "ratio", "split", "sample", "m", "nfe",
          "rmse", "mae", "zstd", "cov95_raw", "width95_raw", "cov95_cal", "width95_cal",
          "t_train", "t_infer", "t_infer_per_draw", "n_missing"]


def read_missflow(d):
    rows = []
    for fp in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(fp) as f:
            rows += list(csv.DictReader(f))
    return rows


def read_diffputer(d):
    """Parse DiffPuter result.txt files -> one row per (dataset,mask,split,sample).
    Path layout: results/<dataset>/rate<r>/<mask>/<split>/<trials_steps>/result.txt.
    Keeps the LAST iteration (EM converged)."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(d, "**", "result.txt"), recursive=True)):
        parts = fp.replace("\\", "/").split("/")
        try:
            i = parts.index("results")
            dataset, rate, mask, split = parts[i + 1], parts[i + 2], parts[i + 3], parts[i + 4]
            ratio = rate.replace("rate", "")
        except (ValueError, IndexError):
            dataset = rate = mask = split = ratio = ""
        mae_in = rmse_in = mae_out = rmse_out = None
        for line in open(fp):
            m = re.search(r"MAE: in-sample: ([\d.]+), out-of-sample: ([\d.]+)", line)
            if m: mae_in, mae_out = float(m.group(1)), float(m.group(2))
            m = re.search(r"RMSE: in-sample: ([\d.]+), out-of-sample: ([\d.]+)", line)
            if m: rmse_in, rmse_out = float(m.group(1)), float(m.group(2))
        uq = {}
        for tag in ("insample", "oos"):
            jp = os.path.join(os.path.dirname(fp), f"uq_{tag}.json")
            if os.path.exists(jp):
                uq[tag] = json.load(open(jp))
        for sample, mae, rmse, tag in [("in_sample", mae_in, rmse_in, "insample"),
                                       ("out_of_sample", mae_out, rmse_out, "oos")]:
            if mae is None:
                continue
            row = dict(dataset=dataset, method="DiffPuter", mask=mask, ratio=ratio,
                       split=split, sample=sample, mae=mae, rmse=rmse)
            row.update(uq.get(tag, {}))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missflow", default="results/missflow")
    ap.add_argument("--diffputer", default="DiffPuter/results")
    ap.add_argument("--out", default="results/comparison.csv")
    a = ap.parse_args()

    rows = []
    if os.path.isdir(a.missflow): rows += read_missflow(a.missflow)
    if os.path.isdir(a.diffputer): rows += read_diffputer(a.diffputer)
    if not rows:
        raise SystemExit("no results found (check --missflow / --diffputer paths)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {len(rows)} rows -> {a.out}")
    # quick console summary: mean RMSE / coverage by (method, sample)
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r.get("method", ""), r.get("sample", ""))
        for k in ("rmse", "cov95_raw", "cov95_cal"):
            v = r.get(k, "")
            if v not in ("", None):
                agg[key][k].append(float(v))
    print(f"\n{'method':<11}{'sample':<15}{'RMSE':>8}{'cov_raw':>9}{'cov_cal':>9}")
    print("-" * 52)
    for (meth, samp), d in sorted(agg.items()):
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"{meth:<11}{samp:<15}{mean(d['rmse']):>8.3f}"
              f"{mean(d['cov95_raw']):>9.3f}{mean(d['cov95_cal']):>9.3f}")


if __name__ == "__main__":
    main()
