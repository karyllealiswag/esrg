"""
run_batch.py — Evaluate a configuration over a BRISC split.

Purpose : Produce per-image scores and a summary for one method/config.
Function : evaluate_split() runs the pipeline over every image/mask pair (optionally a
          stratified subset), writes a per-image CSV, and prints mean/median with
          per-class and per-plane breakdowns.
Notes   : CLI: --root <split> --method esrg|srg [--limit N]. Tuning must use TRAIN;
          report on TEST once.
"""
import argparse
import csv
import os
import random
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg.io_utils import list_pairs, parse_brisc_name
from esrg.metrics import summarize

FIELDS = ["file", "tumor", "plane", "status", "dsc", "iou", "precision", "recall",
          "hd95", "assd", "pred_area", "gt_area", "area_ratio", "leaked",
          "seed_hit", "seconds"]


def evaluate_split(root, cfg, limit=None, seed=0, stratify=True, quiet=False):
    """Run cfg over the split and return (rows, summary)."""
    pairs = [(i, m) for i, m in list_pairs(root) if m]
    if limit and limit < len(pairs):
        random.seed(seed)
        if stratify:                       # equal share per tumor × plane cell
            cells = {}
            for p in pairs:
                k = parse_brisc_name(p[0])
                cells.setdefault((k["tumor"], k["plane"]), []).append(p)
            per = max(1, limit // max(len(cells), 1))
            pairs = [x for v in cells.values() for x in random.sample(v, min(per, len(v)))]
        else:
            pairs = random.sample(pairs, limit)

    rows = []
    for n, (img_path, mask_path) in enumerate(sorted(pairs), 1):
        try:
            res = run(img_path, cfg, mask_path=mask_path)
            meta, sc = res.meta, res.scores
            rows.append({"file": os.path.basename(img_path), "tumor": meta.get("tumor"),
                         "plane": meta.get("plane"), "status": res.status,
                         **{k: sc.get(k) for k in FIELDS if k in sc}})
        except Exception as e:
            rows.append({"file": os.path.basename(img_path), "status": f"ERROR: {e}"})
        if not quiet and n % 25 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)
    return rows, summarize(rows)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def by_group(rows, key):
    """Median DSC and mean recall/precision per tumor class or plane."""
    out = {}
    for g in sorted({r.get(key) for r in rows if r.get(key)}):
        sub = [r for r in rows if r.get(key) == g]
        out[g] = {"n": len(sub), **{k: v["median"] for k, v in
                                    summarize(sub, ("dsc", "precision", "recall")).items()}}
    return out


def print_summary(title, rows, summary):
    ok = [r for r in rows if r.get("status") == "OK"]
    nocand = sum(1 for r in rows if r.get("status") == "NO TUMOR CANDIDATE")
    print(f"\n{title}  (n={len(rows)}, OK={len(ok)}, no-candidate={nocand})")
    for k, v in summary.items():
        print(f"  {k:<10} mean {v['mean']:.3f} ± {v['sd']:.3f}   "
              f"median {v['median']:.3f} [{v['q1']:.3f}–{v['q3']:.3f}]")
    hits = [r["seed_hit"] for r in rows if r.get("seed_hit") is not None]
    if hits:
        print(f"  seed hit rate: {100 * sum(hits) / len(hits):.1f}%  (n={len(hits)})")
    leaks = [r["leaked"] for r in rows if r.get("leaked") is not None]
    if leaks:
        print(f"  leakage rate : {100 * sum(leaks) / len(leaks):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="split folder containing images/ and masks/")
    ap.add_argument("--method", default="esrg", choices=["esrg", "srg"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/results.csv")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    cfg = cfg.replace(method=args.method)

    rows, summary = evaluate_split(args.root, cfg, args.limit)
    write_csv(rows, args.out)
    print_summary(f"{args.method.upper()} on {args.root}", rows, summary)
    print("\nBy tumor class:", by_group(rows, "tumor"))
    print("By plane:", by_group(rows, "plane"))
    print(f"\nPer-image results written to {args.out}")
