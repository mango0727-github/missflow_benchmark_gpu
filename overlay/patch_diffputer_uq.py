"""Additively patch DiffPuter's main.py so it also reports our per-cell coverage.

DiffPuter samples `num_trials` draws then averages them and reports only RMSE/MAE.
We insert, right before each averaging line, a block that runs uq_eval on the FULL draw
stack and writes uq_insample.json / uq_oos.json next to result.txt. Idempotent; matches
the line by stripped content so it is robust to indentation. Run once after cloning:

    python patch_diffputer_uq.py DiffPuter/main.py      (uq_eval.py must be in DiffPuter/)
"""
import sys

TARGET = "rec_X = torch.stack(rec_Xs, dim = 0).mean(0)"


def block(tag, indent):
    X = "X" if tag == "insample" else "X_test"
    num = "train_num" if tag == "insample" else "test_num"
    msk = "train_mask" if tag == "insample" else "test_mask"
    out = f"uq_{tag}.json"
    lines = [
        "# --- MissFlow UQ overlay (added by patch_diffputer_uq.py) ---",
        "try:",
        "    from uq_eval import evaluate_uq as _euq",
        "    import os as _os, json as _json",
        "    _stack = torch.stack(rec_Xs, dim=0).detach().cpu().numpy() * 2.0",
        f"    _Xtrue = {X}.detach().cpu().numpy() * 2.0",
        f"    _nn = {num}.shape[1]",
        f"    _miss = np.asarray({msk})[:, :_nn].astype(bool)",
        "    _uq = _euq(_stack, _Xtrue, _miss, _nn, level=0.95)",
        "    _sp = f'results/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'",
        "    _os.makedirs(_sp, exist_ok=True)",
        f"    _json.dump(_uq, open(f'{{_sp}}/{out}', 'w'))",
        f"    print('UQ {tag}:', _uq)",
        "except Exception as _e:",
        f"    print('UQ overlay ({tag}) skipped:', _e)",
    ]
    return [indent + ln for ln in lines]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    src = open(path).read().split("\n")
    if any("evaluate_uq" in ln for ln in src):
        print(f"{path}: already patched"); return
    out, n = [], 0
    for line in src:
        if line.strip() == TARGET:
            n += 1
            indent = line[:len(line) - len(line.lstrip())]
            out += block("insample" if n == 1 else "oos", indent)
        out.append(line)
    if n == 0:
        print(f"{path}: target line not found - DiffPuter main.py may have changed."); sys.exit(2)
    open(path, "w").write("\n".join(out))
    print(f"{path}: patched {n} sampling loop(s) with UQ.")


if __name__ == "__main__":
    main()
