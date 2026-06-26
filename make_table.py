"""Final paper comparison table: MissFlow (our harness, reproduction-validated) vs the
diffusion imputers' PUBLISHED numbers (DiffPuter ICLR'25 Table 8). MissFlow numbers are
averaged over splits from our run; baseline numbers come from published_reference.csv.
The harness's reproduction was validated separately (TabCSDI reproduces; DiffPuter's EM
converges to its published RMSE), which is what makes comparing to published valid.

  python make_table.py --reproduced results/comparison.csv --reference published_reference.csv
  python make_table.py --baselines DiffPuter,TabCSDI,MissDiff   # include MissDiff too

Default baselines are DiffPuter,TabCSDI (MissDiff is a high-variance baseline, handled
separately). Prints console + Markdown + LaTeX; lowest RMSE per dataset is marked.
"""
from __future__ import annotations
import argparse, csv, os
from collections import defaultdict

NFE = {"MissFlow": 20, "DiffPuter": 50, "TabCSDI": 100, "MissDiff": 50}


def _mean(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return sum(xs) / len(xs) if xs else None


def load_run(fp, sample):
    """All methods we actually ran: {method: {dataset: mean_rmse}} + MissFlow coverage."""
    rmse, cov = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for r in csv.DictReader(open(fp)):
        if r.get("sample") != sample:
            continue
        m = r.get("method", "")
        if r.get("rmse"): rmse[m][r["dataset"]].append(r["rmse"])
        if m == "MissFlow" and r.get("cov95_cal"): cov[r["dataset"]].append(r["cov95_cal"])
    run = {m: {d: _mean(v) for d, v in dd.items()} for m, dd in rmse.items()}
    return run, {d: _mean(v) for d, v in cov.items()}


def load_published(fp, sample):
    out = defaultdict(dict)
    for r in csv.DictReader(open(fp)):
        if r.get("metric") != "rmse" or (r.get("sample") or "in_sample") != sample:
            continue
        v = (r.get("value", "") or "").strip()
        if v: out[r["method"]][r["dataset"]] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduced", default="results/comparison.csv")
    ap.add_argument("--reference", default="published_reference.csv")
    ap.add_argument("--sample", default="in_sample")
    ap.add_argument("--baselines", default="DiffPuter,TabCSDI,MissDiff")
    ap.add_argument("--prefer-run", action="store_true",
                    help="use OUR run for a baseline if we ran it (default: always published)")
    ap.add_argument("--out", default="results/comparison_table.md")
    a = ap.parse_args()
    baselines = [b for b in a.baselines.split(",") if b]

    run, mf_cov = load_run(a.reproduced, a.sample)
    pub = load_published(a.reference, a.sample)
    datasets = sorted(set(run.get("MissFlow", {})) | {d for b in baselines for d in pub.get(b, {})})
    methods = ["MissFlow"] + baselines

    nfe_str = ", ".join(f"{m} {NFE.get(m,'?')}" for m in methods)
    src = "OUR run if present, else published" if a.prefer_run else "DiffPuter ICLR'25 published"
    header = (f"Imputation RMSE ({a.sample}, MAR 30%). MissFlow = ours (reproduction-validated "
              f"harness); baselines = {src}. NFE: {nfe_str}.")

    def rmse_of(m, ds):
        if m == "MissFlow":
            return run.get(m, {}).get(ds)
        if a.prefer_run:                      # opt-in: use our run for a baseline if we ran it
            v = run.get(m, {}).get(ds)
            if v is not None: return v
        return pub.get(m, {}).get(ds)         # default: the published value

    rows = []
    for ds in datasets:
        vals = {m: rmse_of(m, ds) for m in methods}
        numeric = {k: v for k, v in vals.items() if v is not None}
        best = min(numeric, key=numeric.get) if numeric else None
        rows.append((ds, vals, mf_cov.get(ds), best))

    def fnum(v): return f"{v:.3f}" if v is not None else "--"

    def cell(v, bold):
        if v is None: return "--"
        return f"**{v:.3f}**" if bold else f"{v:.3f}"

    # ---- console ----
    print("\n" + header + "\n")
    head = f"{'dataset':<12}{'MissFlow':>10}{'cov':>6}" + "".join(f"{m:>11}" for m in baselines)
    print(head); print("-" * len(head))
    for ds, vals, cov, best in rows:
        covs = f"{cov:.2f}" if cov is not None else "-"
        line = f"{ds:<12}{fnum(vals['MissFlow']):>10}{covs:>6}"
        line += "".join(f"{fnum(vals[b]):>11}" for b in baselines)
        print(line + ("   <- MissFlow best" if best == "MissFlow" else ""))

    # ---- markdown + LaTeX ----
    md = [f"<!-- {header} -->", "",
          "| Dataset | MissFlow | cov95 | " + " | ".join(baselines) + " |",
          "|---" * (3 + len(baselines)) + "|"]
    tex = ["\\begin{tabular}{lcc" + "c" * len(baselines) + "}", "\\toprule",
           "Dataset & MissFlow & cov$_{95}$ & " + " & ".join(baselines) + " \\\\", "\\midrule"]
    for ds, vals, cov, best in rows:
        covs = f"{cov:.2f}" if cov is not None else "--"
        md.append(f"| {ds} | {cell(vals['MissFlow'], best=='MissFlow')} | {covs} | "
                  + " | ".join(cell(vals[b], best == b) for b in baselines) + " |")
        tcell = lambda m: ("--" if vals[m] is None else
                           (f"\\textbf{{{vals[m]:.3f}}}" if best == m else f"{vals[m]:.3f}"))
        tex.append(f"{ds} & {tcell('MissFlow')} & {covs} & "
                   + " & ".join(tcell(b) for b in baselines) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(md) + "\n\n```latex\n" + "\n".join(tex) + "\n```\n")
    print(f"\nwrote markdown + LaTeX -> {a.out}")


if __name__ == "__main__":
    main()
