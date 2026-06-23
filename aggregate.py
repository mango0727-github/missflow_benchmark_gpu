"""Merge MissFlow + all diffusion baselines (DiffPuter, MissDiff, TabCSDI) into ONE
comparison CSV. Each baseline writes its own result format under the DiffPuter clone:

  DiffPuter : <root>/results/<ds>/rate<r>/<mask>/<split>/<trials_steps>/result.txt
  MissDiff  : <root>/baselines/Missdiff_SDE/missdiff_sde_output/<mask>/eval_result/<ds>/<init>/result_mask<idx>.txt
  TabCSDI   : <root>/baselines/TabCSDI/TabCSDI_output/<mask>/{in,out}_sample_result/<ds>/*_result_mask<idx>.txt
  MissFlow  : <missflow_dir>/missflow_*.csv  (our runner; already has accuracy + UQ)

Accuracy is the same metric for all (each baseline uses DiffPuter-style get_eval,
standardized-space RMSE/MAE on missing cells), so rows are directly comparable. Coverage
columns are present only for methods that emit a draw stack (MissFlow always; diffusion
baselines if the UQ patch saved uq_*.json).

  python aggregate.py --missflow results/missflow --diffputer-root DiffPuter --out results/comparison.csv
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re

FIELDS = ["dataset", "method", "mask", "split", "sample", "rmse", "mae", "nfe",
          "zstd", "cov95_raw", "width95_raw", "cov95_cal", "width95_cal",
          "m", "t_train", "t_infer", "t_infer_per_draw", "n_missing"]
NFE = {"MissFlow": 20, "DiffPuter": 50, "MissDiff": 50, "TabCSDI": 100}


def _norm_mask(m):                                    # unify the MNAR naming
    return "MNAR" if m and m.startswith("MNAR") else m


def read_missflow(d):
    rows = []
    for fp in sorted(glob.glob(os.path.join(d, "*.csv"))):
        rows += list(csv.DictReader(open(fp)))
    for r in rows:
        r["mask"] = _norm_mask(r.get("mask", ""))
    return rows


def _attach_uq(folder, tag, row):
    jp = os.path.join(folder, f"uq_{tag}.json")
    if os.path.exists(jp):
        row.update(json.load(open(jp)))


def read_diffputer(root):
    rows = []
    for fp in glob.glob(os.path.join(root, "results", "**", "result.txt"), recursive=True):
        p = fp.replace("\\", "/").split("/")
        try:
            i = p.index("results"); ds, mask, split = p[i + 1], p[i + 3], p[i + 4]
        except (ValueError, IndexError):
            continue
        vals = {}
        for line in open(fp):
            for met, key in (("MAE", "mae"), ("RMSE", "rmse")):
                m = re.search(rf"{met}: in-sample: ([\d.]+), out-of-sample: ([\d.]+)", line)
                if m: vals[(key, "in_sample")], vals[(key, "out_of_sample")] = float(m[1]), float(m[2])
        for sample in ("in_sample", "out_of_sample"):
            if ("rmse", sample) not in vals: continue
            row = dict(dataset=ds, method="DiffPuter", mask=_norm_mask(mask), split=split,
                       sample=sample, rmse=vals.get(("rmse", sample)), mae=vals.get(("mae", sample)))
            _attach_uq(os.path.dirname(fp), "insample" if sample == "in_sample" else "oos", row)
            rows.append(row)
    return rows


def read_missdiff(root):
    rows = []
    base = os.path.join(root, "baselines", "Missdiff_SDE", "missdiff_sde_output")
    for fp in glob.glob(os.path.join(base, "**", "result_mask*.txt"), recursive=True):
        p = fp.replace("\\", "/").split("/")
        idx = re.search(r"result_mask(\d+)\.txt", fp); split = idx[1] if idx else ""
        try:
            i = p.index("missdiff_sde_output"); mask = p[i + 1]; ds = p[i + 3]
        except (ValueError, IndexError):
            continue
        vals = {}
        for line in open(fp):
            for met, key in (("MAE", "mae"), ("RMSE", "rmse")):
                m = re.search(rf"{met}: in-sample: ([\d.]+), out-of-sample: ([\d.]+)", line)
                if m: vals[(key, "in_sample")], vals[(key, "out_of_sample")] = float(m[1]), float(m[2])
        for sample in ("in_sample", "out_of_sample"):
            if ("rmse", sample) not in vals: continue
            rows.append(dict(dataset=ds, method="MissDiff", mask=_norm_mask(mask), split=split,
                             sample=sample, rmse=vals.get(("rmse", sample)), mae=vals.get(("mae", sample))))
    return rows


def read_tabcsdi(root):
    rows = []
    base = os.path.join(root, "baselines", "TabCSDI", "TabCSDI_output")
    pat = re.compile(r"TabCSDI (?:in|out) sample RMSE: ([\d.]+)")
    for kind, sample in (("in_sample_result", "in_sample"), ("out_sample_result", "out_of_sample")):
        for fp in glob.glob(os.path.join(base, "*", kind, "*", "*_result_mask*.txt")):
            p = fp.replace("\\", "/").split("/")
            try:
                i = p.index("TabCSDI_output"); mask = p[i + 1]; ds = p[i + 3]
            except (ValueError, IndexError):
                continue
            idx = re.search(r"_result_mask(\d+)\.txt", fp); split = idx[1] if idx else ""
            rmse = next((float(pat.search(l)[1]) for l in open(fp) if pat.search(l)), None)
            if rmse is None: continue
            rows.append(dict(dataset=ds, method="TabCSDI", mask=_norm_mask(mask), split=split,
                             sample=sample, rmse=rmse))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missflow", default="results/missflow")
    ap.add_argument("--diffputer-root", default="DiffPuter")
    ap.add_argument("--out", default="results/comparison.csv")
    a = ap.parse_args()

    rows = []
    if os.path.isdir(a.missflow): rows += read_missflow(a.missflow)
    root = a.diffputer_root
    if os.path.isdir(root):
        rows += read_diffputer(root) + read_missdiff(root) + read_tabcsdi(root)
    if not rows:
        raise SystemExit("no results found (check --missflow / --diffputer-root)")

    for r in rows:                                    # fill known NFE if missing
        r.setdefault("nfe", NFE.get(r.get("method", ""), ""))
        if not r.get("nfe"): r["nfe"] = NFE.get(r.get("method", ""), "")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {len(rows)} rows -> {a.out}")

    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("rmse", "cov95_raw", "cov95_cal"):
            v = r.get(k, "")
            if v not in ("", None): agg[(r["method"], r["sample"])][k].append(float(v))
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"\n{'method':<11}{'sample':<15}{'nfe':>5}{'RMSE':>8}{'cov_raw':>9}{'cov_cal':>9}")
    print("-" * 57)
    for (meth, samp), d in sorted(agg.items()):
        print(f"{meth:<11}{samp:<15}{NFE.get(meth,''):>5}{mean(d['rmse']):>8.3f}"
              f"{mean(d['cov95_raw']):>9.3f}{mean(d['cov95_cal']):>9.3f}")


if __name__ == "__main__":
    main()
