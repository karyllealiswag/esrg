"""
ablation.py — Experiment E4: contribution of each enhancement.

Purpose : Isolate the effect of the local log measure and the stopping criterion.
Function : Runs five arms (SRG; global-delta with/without stop; local-delta with/
          without stop; full ESRG) with seeds held fixed, and prints a comparison.
Notes   : CLI: --root <split> [--limit N]. Writes one CSV per arm.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrg import Config
from experiments.run_batch import evaluate_split, write_csv

ARMS = [("SRG baseline",            dict(method="srg")),
        ("Global delta, no stop",   dict(use_log_local=False, use_stopping=False)),
        ("Global delta + stop",     dict(use_log_local=False, use_stopping=True)),
        ("Local log delta, no stop", dict(use_log_local=True,  use_stopping=False)),
        ("ESRG (full)",             dict(use_log_local=True,  use_stopping=True))]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="outputs/ablation")
    args = ap.parse_args()

    print(f"{'arm':<26}{'DSC med':>9}{'DSC mean':>10}{'prec':>8}{'recall':>8}{'leak%':>7}")
    for name, over in ARMS:
        cfg = Config().replace(**over)
        rows, s = evaluate_split(args.root, cfg, args.limit, quiet=True)
        write_csv(rows, os.path.join(args.out, name.replace(" ", "_") + ".csv"))
        lk = [r["leaked"] for r in rows if r.get("leaked") is not None]
        print(f"{name:<26}{s['dsc']['median']:>9.3f}{s['dsc']['mean']:>10.3f}"
              f"{s['precision']['mean']:>8.3f}{s['recall']['mean']:>8.3f}"
              f"{100 * sum(lk) / max(len(lk), 1):>7.1f}")
