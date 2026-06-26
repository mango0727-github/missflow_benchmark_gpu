"""Final paper comparison table: MissFlow (our harness, reproduction-validated) vs the
diffusion imputers' PUBLISHED numbers (DiffPuter ICLR'25 Table 8). MissFlow numbers are
averaged over splits from our run; baseline numbers are read from published_reference.csv.
The harness's reproduction was validated separately (TabCSDI reproduces; DiffPuter's EM
converges to its published RMSE), which is what makes comparing to published numbers valid.

  python make_table.py --reproduced results/comparison.csv --reference published_reference.csv

Prints a console + Markdown + LaTeX table (rows = datasets; MissFlow RMSE/coverage + each
baseline's published RMSE; NFE in the header). Lowest RMSE per row is marked.
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict

NFE = {"MissFlow": 20, "DiffPuter": 50, "TabCSDI": 100, "MissDiff": 50}
BASELINES = ["DiffPuter", "TabCSDI", "MissDiff"]


def _mean(xs):
    xs = [x for x in xs if x not in ("", None)]
    return sum(map(float, xs)) / len(xs) if xs else None


def load_missflow(fp, sample):
    rmse, cov = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(fp)):
        if r.get("method") != "MissFlow" or r.get("sample") != sample:
            continue
        if r.get("rmse"): rmse[r["dataset"]].append(r["rmse"])
        if r.get("cov95_cal"): cov[r["dataset"]].append(r["cov95_cal"])
    return ({d: _mean(v) for d, v in rmse.items()},
            {d: _mean(v) for d, v in cov.items()})


def load_published(fp, sample):
    out = defaultdict(dict)
    for r in csv.DictReader(open(fp)):
        if r.get("metric") != "rmse" or (r.get("sample") or "in_sample") != sample:
            continue
        v = (r.get("value", "") or "").strip()
        if v: out[r["method"]][r["dataset"]] = float(v)
    return out


def fmt(x, best=False):
    if x is None: return "  -  "
    s = f"{x:.3f}"
    return f"**{s}**" if best else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduced", default="results/comparison.csv")
    ap.add_argument("--reference", default="published_reference.csv")
    ap.add_argument("--sample", default="in_sample")
    ap.add_argument("--out", default="results/comparison_table.md")
    a = ap.parse_args()

    mf_rmse, mf_cov = load_missflow(a.reproduced, a.sample)
    pub = load_published(a.reference, a.sample)
    datasets = sorted(set(mf_rmse) | {d for m in pub.values() for d in m})

    header = (f"Imputation RMSE ({a.sample}, MAR 30%). MissFlow = ours (reproduction-validated "
              f"harness); baselines = DiffPuter ICLR'25 published. NFE: MissFlow {NFE['MissFlow']}, "
              f"DiffPuter {NFE['DiffPuter']}, TabCSDI {NFE['TabCSDI']}, MissDiff {NFE['MissDiff']}.")
    cols = ["dataset", "MissFlow", "cov95", "DiffPuter", "TabCSDI", "MissDiff"]

    rows = []
    for ds in datasets:
        vals = {"MissFlow": mf_rmse.get(ds)}
        for b in BASELINES:
            vals[b] = pub.get(b, {}).get(ds)
        numeric = {k: v for k, v in vals.items() if v is not None}
        best = min(numeric, key=numeric.get) if numeric else None
        rows.append((ds, vals, mf_cov.get(ds), best))

    # ---- console + markdown ----
    md = [f"<!-- {header} -->", "",
          "| Dataset | MissFlow | cov95 | DiffPuter | TabCSDI | MissDiff |",
          "|---|---|---|---|---|---|"]
    print("\n" + header + "\n")
    print(f"{'dataset':<12}{'MissFlow':>10}{'cov':>7}{'DiffPuter':>11}{'TabCSDI':>9}{'MissDiff':>10}")
    print("-" * 60)
    for ds, vals, cov, best in rows:
        covs = f"{cov:.2f}" if cov is not None else "-"
        print(f"{ds:<12}{fmt(vals['MissFlow']).replace('*',''):>10}{covs:>7}"
              f"{fmt(vals['DiffPuter']).replace('*',''):>11}{fmt(vals['TabCSDI']).replace('*',''):>9}"
              f"{fmt(vals['MissDiff']).replace('*',''):>10}" + ("   (MissFlow best)" if best == "MissFlow" else ""))
        md.append(f"| {ds} | {fmt(vals['MissFlow'], best=='MissFlow')} | {covs} | "
                  f"{fmt(vals['DiffPuter'], best=='DiffPuter')} | {fmt(vals['TabCSDI'], best=='TabCSDI')} | "
                  f"{fmt(vals['MissDiff'], best=='MissDiff')} |")

    # ---- LaTeX ----
    tex = ["\\begin{tabular}{lccccc}", "\\toprule",
           "Dataset & MissFlow & cov$_{95}$ & DiffPuter & TabCSDI & MissDiff \\\\",
           "\\midrule"]
    for ds, vals, cov, best in rows:
        def t(k):
            v = vals[k]; s = "--" if v is None else f"{v:.3f}"
            return f"\\textbf{{{s}}}" if best == k and v is not None else s
        covs = "--" if cov is None else f"{cov:.2f}"
        tex.append(f"{ds} & {t('MissFlow')} & {covs} & {t('DiffPuter')} & {t('TabCSDI')} & {t('MissDiff')} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n\n```latex\n" + "\n".join(tex) + "\n```\n")
    print(f"\nwrote markdown + LaTeX -> {a.out}")


if __name__ == "__main__":
    main()
