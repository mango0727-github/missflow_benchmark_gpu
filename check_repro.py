"""Reproduction gate: does our harness reproduce each paper's PUBLISHED numbers?

If the official baselines (DiffPuter, TabCSDI, MissDiff) run through this harness match
their published RMSE, then (a) our setup is correct and (b) MissFlow's numbers on the
SAME harness are credible. This is the check that must pass before the comparison means
anything.

  python check_repro.py --reproduced results/comparison.csv \
                        --reference published_reference.csv --tol 0.10

published_reference.csv columns: method,dataset,mask,sample,metric,value,source
(metric = rmse|mae ; sample = in_sample|out_of_sample ; value read from the paper table).
Reproduced numbers are averaged over splits before comparing.
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict


def load_reference(fp):
    ref = {}
    for r in csv.DictReader(open(fp)):
        v = (r.get("value", "") or "").strip()
        if v in ("", "NA", "nan", "TODO"):
            continue
        key = (r["method"], r["dataset"], r.get("mask", "MCAR") or "MCAR",
               r.get("sample", "in_sample") or "in_sample", r.get("metric", "rmse") or "rmse")
        ref[key] = (float(v), r.get("source", ""))
    return ref


def load_reproduced(fp):
    agg = defaultdict(list)
    for r in csv.DictReader(open(fp)):
        for metric in ("rmse", "mae"):
            v = (r.get(metric, "") or "").strip()
            if v in ("", "nan"):
                continue
            key = (r["method"], r["dataset"], r.get("mask", "MCAR") or "MCAR",
                   r.get("sample", "in_sample") or "in_sample", metric)
            agg[key].append(float(v))
    return {k: sum(v) / len(v) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduced", default="results/comparison.csv")
    ap.add_argument("--reference", default="published_reference.csv")
    ap.add_argument("--tol", type=float, default=0.10, help="relative tolerance to PASS")
    a = ap.parse_args()

    ref, rep = load_reference(a.reference), load_reproduced(a.reproduced)
    if not ref:
        raise SystemExit(f"{a.reference} has no usable numbers yet (fill values from the papers).")

    print(f"{'':4} {'method':<10}{'dataset':<10}{'msk':<6}{'samp':<5}{'met':<5}"
          f"{'published':>10}{'repro':>9}{'Δ%':>7}  source")
    print("-" * 78)
    npass = nfail = nmiss = 0
    for key in sorted(ref):
        method, ds, mask, sample, metric = key
        pub, src = ref[key]
        samp = "in" if sample.startswith("in") else "out"
        if key not in rep:
            nmiss += 1
            print(f"{'----':4} {method:<10}{ds:<10}{mask:<6}{samp:<5}{metric:<5}"
                  f"{pub:>10.3f}{'MISSING':>9}{'':>7}  (not run yet)")
            continue
        got = rep[key]
        rel = abs(got - pub) / pub if pub else float("inf")
        ok = rel <= a.tol
        npass += ok; nfail += not ok
        print(f"{'PASS' if ok else 'FAIL':4} {method:<10}{ds:<10}{mask:<6}{samp:<5}{metric:<5}"
              f"{pub:>10.3f}{got:>9.3f}{rel*100:>6.1f}%  {src}")
    print("-" * 78)
    print(f"{npass} pass / {nfail} fail / {nmiss} not-yet-run   (tol {a.tol*100:.0f}%)")
    if nfail:
        print("FAIL rows = harness does not reproduce the paper there -> fix protocol before trusting.")


if __name__ == "__main__":
    main()
