"""
run_attribution.py — Full-dataset failure attribution and per-class report.

Purpose : Run attribution over a whole split and summarize where the system fails.
Function : evaluate() records stage metrics + bucket per image; report() prints the
          bucket table, the bimodal DSC split, seed-correct-only growth quality, and
          a tumor-class x plane grid.
Notes   : This is the evidence base for Chapter 4. Seed-correct DSC (growth ceiling)
          is reported separately from whole-set DSC (gated by seeding).
"""
import argparse
import collections
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg.io_utils import list_pairs, parse_brisc_name
from experiments.attribution import stage_metrics, attribute, BUCKET_LABEL, SUCCESS_DSC


def _nanmedian(v):
    v = np.array([x for x in v if x is not None and x == x], float)
    return float(np.median(v)) if v.size else float("nan")


def _rate(v, pred):
    v = [x for x in v if x is not None and x == x]
    return 100.0 * sum(1 for x in v if pred(x)) / len(v) if v else float("nan")


def evaluate(root, cfg, limit=None, out_csv=None, progress_every=100):
    pairs = [(i, m) for i, m in list_pairs(root) if m]
    pairs.sort()
    if limit:
        pairs = pairs[:limit]

    rows = []
    for n, (img_path, mask_path) in enumerate(pairs, 1):
        rec = {"file": os.path.basename(img_path), **parse_brisc_name(img_path)}
        try:
            res = run(img_path, cfg, mask_path=mask_path)
            sm = stage_metrics(res)
            bucket, reason = attribute(res)
            rec.update(sm)
            rec["status"] = res.status
            rec["dsc"] = res.scores.get("dsc")
            rec["precision"] = res.scores.get("precision")
            rec["recall"] = res.scores.get("recall")
            rec["hd95"] = res.scores.get("hd95")
            rec["seconds"] = res.scores.get("seconds")
            rec["seed_hit"] = res.scores.get("seed_hit")
            rec["bucket"] = bucket
            rec["reason"] = reason
        except Exception as e:
            rec["status"] = f"ERROR: {e}"
            rec["bucket"] = "?"
        rows.append(rec)
        if progress_every and n % progress_every == 0:
            print(f"  {n}/{len(pairs)}", flush=True)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


def report(rows):
    n = len(rows)
    print(f"\n{'='*70}\nFAILURE ATTRIBUTION  (n={n})\n{'='*70}")
    counts = collections.Counter(r["bucket"] for r in rows)
    for b in ["G", "C", "B", "E", "D", "A", "F", "X", "?"]:
        if counts.get(b):
            print(f"  {b}  {BUCKET_LABEL[b]:<42} {counts[b]:4d}  {100*counts[b]/n:5.1f}%")

    # Overall DSC distribution (the bimodal check)
    dsc = [r.get("dsc") for r in rows]
    print(f"\nDSC distribution (all images):")
    print(f"  median {_nanmedian(dsc):.3f}   "
          f"DSC=0 rate {_rate(dsc, lambda x: x==0):.1f}%   "
          f">=0.7 rate {_rate(dsc, lambda x: x>=SUCCESS_DSC):.1f}%")

    # Seed-correct-only growth ceiling
    ok = [r for r in rows if r.get("s5_seed_in_gt_frac") is not None
          and r["s5_seed_in_gt_frac"] == r["s5_seed_in_gt_frac"]
          and r["s5_seed_in_gt_frac"] >= 0.5]
    if ok:
        d_ok = [r.get("dsc") for r in ok]
        print(f"\nGrowth quality on SEED-CORRECT images only (n={len(ok)}, "
              f"{100*len(ok)/n:.0f}% of set):")
        print(f"  DSC median {_nanmedian(d_ok):.3f}   "
              f">=0.7 rate {_rate(d_ok, lambda x: x>=SUCCESS_DSC):.0f}%   "
              f"<0.3 rate {_rate(d_ok, lambda x: x<0.3):.0f}%")
        # Even with a correct seed the result is bimodal: separate the two modes.
        leak = [r for r in ok if (r.get("precision") or 1) < 0.5 and (r.get("recall") or 0) >= 0.5]
        under = [r for r in ok if (r.get("recall") or 1) < 0.5 and (r.get("precision") or 0) >= 0.5]
        good = [r for r in ok if (r.get("dsc") or 0) >= SUCCESS_DSC]
        print(f"  of these: good {len(good)} ({100*len(good)/len(ok):.0f}%)  "
              f"leak {len(leak)} ({100*len(leak)/len(ok):.0f}%)  "
              f"undersegment {len(under)} ({100*len(under)/len(ok):.0f}%)")
        print("  -> seed-correct results are themselves bimodal; a single k_L fits neither mode.")

    # Per-class x plane
    print(f"\n{'='*70}\nPER CLASS x PLANE\n{'='*70}")
    print(f"{'class':<12}{'plane':<10}{'n':>4}{'DSCmed':>8}{'=0%':>6}{'>=.7%':>7}  bucket mix")
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r.get("tumor"), r.get("plane"))].append(r)
    for (t, p) in sorted(cells, key=lambda k: (str(k[0]), str(k[1]))):
        sub = cells[(t, p)]
        d = [x.get("dsc") for x in sub]
        mix = collections.Counter(x["bucket"] for x in sub)
        mixs = " ".join(f"{b}:{mix[b]}" for b in ["G","C","B","E","D","A","F","X"] if mix.get(b))
        print(f"{str(t):<12}{str(p):<10}{len(sub):>4}{_nanmedian(d):>8.3f}"
              f"{_rate(d, lambda x: x==0):>6.0f}{_rate(d, lambda x: x>=SUCCESS_DSC):>7.0f}  {mixs}")

    # Per-class totals
    print(f"\n{'class':<12}{'n':>5}{'DSCmed':>8}{'DSCmean':>9}{'=0%':>6}{'>=.7%':>7}{'seedhit%':>9}")
    byc = collections.defaultdict(list)
    for r in rows:
        byc[r.get("tumor")].append(r)
    for t in sorted(byc, key=str):
        sub = byc[t]
        d = [x.get("dsc") for x in sub if x.get("dsc") is not None]
        sh = [x.get("s5_seed_in_gt_frac") for x in sub]
        print(f"{str(t):<12}{len(sub):>5}{_nanmedian(d):>8.3f}"
              f"{(np.nanmean(d) if d else float('nan')):>9.3f}"
              f"{_rate([x.get('dsc') for x in sub], lambda x: x==0):>6.0f}"
              f"{_rate([x.get('dsc') for x in sub], lambda x: x>=SUCCESS_DSC):>7.0f}"
              f"{_rate(sh, lambda x: x>=0.5):>9.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", default="esrg", choices=["esrg", "srg"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/attribution.csv")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = (Config.load(args.config) if args.config else Config()).replace(method=args.method)
    rows = evaluate(args.root, cfg, args.limit, args.out)
    report(rows)
    print(f"\nPer-image attribution written to {args.out}")
