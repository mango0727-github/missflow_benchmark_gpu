"""Rewrite the hardcoded dataset / mask / split loops in a DiffPuter baseline runner so
it benchmarks only OUR subset, instead of every dataset x mask x 10 splits.

TabCSDI's csdi_benchmark.py and MissDiff's Missdiff_benchmark.py both loop internally over
hardcoded `datanames`/`mask_types` lists and `range(mask_num)`; this edits those in place.
Idempotent (re-running just re-applies the same subset).

  python set_subset.py FILE --datasets "magic bean letter shoppers" --masks MCAR --splits 3
  python set_subset.py FILE --epochs 200      # also shrink num_epochs (for a smoke)
"""
import argparse, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--datasets", default="magic bean letter shoppers")
    ap.add_argument("--masks", default="MCAR")
    ap.add_argument("--splits", type=int, default=3, help="use splits 0..splits-1")
    ap.add_argument("--epochs", type=int, default=0, help="if >0, rewrite num_epochs default")
    a = ap.parse_args()

    src = open(a.file).read()
    dl = "[" + ", ".join(f"'{d}'" for d in a.datasets.split()) + "]"
    ml = "[" + ", ".join(f"'{m}'" for m in a.masks.split()) + "]"
    n = 0
    src, k = re.subn(r"datanames\s*=\s*\[[^\]]*\]", f"datanames = {dl}", src); n += k
    src, k = re.subn(r"mask_types\s*=\s*\[[^\]]*\]", f"mask_types = {ml}", src); n += k
    src, k = re.subn(r"for\s+mask_type\s+in\s+\[[^\]]*\]", f"for mask_type in {ml}", src); n += k
    src, k = re.subn(r"range\(\s*(?:args\.)?mask_num\s*\)", f"range({a.splits})", src); n += k
    if a.epochs > 0:
        # only numeric assignments (e.g. `num_epochs = 10000 + 1`), not `= args.num_epochs`
        src, k = re.subn(r"num_epochs\s*=\s*\d[\d+ ]*", f"num_epochs = {a.epochs}", src); n += k
        src, k = re.subn(r"(--num_epochs[^\n]*default=)\d[\d+ ]*", rf"\g<1>{a.epochs}", src); n += k
    open(a.file, "w").write(src)
    print(f"{a.file}: rewrote {n} loop(s) -> datasets={a.datasets} masks={a.masks} "
          f"splits=0..{a.splits-1}" + (f" epochs={a.epochs}" if a.epochs else ""))


if __name__ == "__main__":
    main()
