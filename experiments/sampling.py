"""
sampling.py — Sample-size determination and equal-allocation stratified sampling.

Purpose : Draw the evaluation sample of Chapter 4 from the pooled BRISC 2025
          segmentation slices (train + test) and record it in a manifest, so every
          later step reads the same slices even after the unused files are removed.
Function : cochran() gives the minimum sample for a proportion with the finite
          population correction; plan() applies it to each tumor class (the level at
          which results are reported) and spreads the largest class requirement
          equally over the three imaging planes, giving the same n_h in all nine
          tumor-class x plane strata; draw() selects n_h slices per stratum by simple
          random sampling without replacement (fixed seed); prune() deletes every
          image/mask pair that is not in the manifest.
Notes   : The manifest (sample.csv) and the population frame (frame.csv, every
          slice with its random draw order and a selected flag) are the record of
          the draw: once the unused files are pruned, the population can no longer
          be re-listed from disk, so draw() refuses to overwrite an existing manifest
          unless --force is given.
          CLI: python experiments/sampling.py --draw            (write manifest)
               python experiments/sampling.py --prune --yes     (delete unused files)
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrg.io_utils import list_pairs, parse_brisc_name

SPLITS = ["segmentation_task/train", "segmentation_task/test"]
OUT = "outputs/evaluation"
SEED = 2026                    # random seed of the draw (reproducible sample)
Z95 = 1.959964                 # standard normal quantile for 95% confidence
MARGIN = 0.05                  # tolerated margin of error of a proportion
P0 = 0.5                       # assumed proportion (most conservative)
CLASSES = ["glioma", "meningioma", "pituitary"]
PLANES = ["axial", "coronal", "sagittal"]


# ─── Sample size ─────────────────────────────────────────────────────────────
def cochran(N, z=Z95, e=MARGIN, p=P0):
    """Cochran (1977): n0 = z^2 p(1-p) / e^2, then n = n0 / (1 + (n0 - 1) / N)."""
    n0 = z * z * p * (1 - p) / (e * e)
    n = n0 / (1 + (n0 - 1) / N)
    return {"N": N, "n0": n0, "n_exact": n, "n_min": math.ceil(n)}


def margin(n, N, z=Z95, p=P0):
    """Achieved margin of error of a proportion: e = z sqrt(p(1-p)/n * (N-n)/(N-1))."""
    return z * math.sqrt(p * (1 - p) / n * (N - n) / (N - 1))


def population(splits=SPLITS):
    """Every (image, mask) pair of the pooled splits, grouped by (tumor, plane) stratum."""
    cells = {}
    for root in splits:
        for ip, mp in list_pairs(root):
            if not mp:
                continue
            k = parse_brisc_name(ip)
            cells.setdefault((k["tumor"], k["plane"]), []).append((ip, mp))
    return cells


def plan(cells):
    """
    Per-class Cochran requirement -> equal allocation over the nine strata.
    n_class = max_c ceil(Cochran(N_c)); n_h = ceil(n_class / 3); n = 9 n_h.
    """
    N_c = {c: sum(len(v) for (t, _), v in cells.items() if t == c) for c in CLASSES}
    per_class = {c: cochran(N_c[c]) for c in CLASSES}
    n_class = max(r["n_min"] for r in per_class.values())
    n_h = math.ceil(n_class / len(PLANES))
    N = sum(N_c.values())
    n = n_h * len(CLASSES) * len(PLANES)
    smallest = min(len(v) for v in cells.values())
    assert n_h <= smallest, f"n_h = {n_h} exceeds the smallest stratum ({smallest})"
    return {"N": N, "N_c": N_c, "per_class": per_class, "n_class_required": n_class,
            "n_h": n_h, "n": n, "overall": cochran(N),
            "margin_overall": margin(n, N),
            "margin_class": {c: margin(n_h * len(PLANES), N_c[c]) for c in CLASSES}}


# ─── Draw ────────────────────────────────────────────────────────────────────
def draw(cells, n_h, seed=SEED):
    """
    Simple random sampling without replacement inside each stratum: the stratum's
    slices are sorted by name, shuffled with a fixed seed, and the first n_h are
    taken. Returns (sample rows, frame rows) — the frame keeps every slice with its
    shuffled position so the selection can be audited.
    """
    rng = random.Random(seed)
    sample, frame = [], []
    for h in sorted(cells):
        items = sorted(cells[h])
        order = list(range(len(items)))
        rng.shuffle(order)
        for pos, i in enumerate(order):
            ip, mp = items[i]
            k = parse_brisc_name(ip)
            row = {"file": os.path.basename(ip), "split": k["split"], "index": k["index"],
                   "tumor": k["tumor"], "plane": k["plane"], "stratum": f"{k['tumor']}|{k['plane']}",
                   "draw_order": pos + 1, "selected": pos < n_h,
                   "image": os.path.relpath(ip).replace("\\", "/"),
                   "mask": os.path.relpath(mp).replace("\\", "/")}
            frame.append(row)
            if pos < n_h:
                sample.append(row)
    return sorted(sample, key=lambda r: r["file"]), frame


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_manifest(out=OUT, force=False, seed=SEED):
    path = os.path.join(out, "sample.csv")
    if os.path.isfile(path) and not force:
        raise SystemExit(f"{path} exists; the sample is fixed. Use --force to redraw "
                         "(only possible while the full population is still on disk).")
    cells = population()
    p = plan(cells)
    sample, frame = draw(cells, p["n_h"], seed)
    os.makedirs(out, exist_ok=True)
    fields = ["file", "split", "index", "tumor", "plane", "stratum", "draw_order",
              "selected", "image", "mask"]
    _write(path, sample, fields)
    _write(os.path.join(out, "frame.csv"), frame, fields)
    strata = [{"tumor": t, "plane": pl, "N_h": len(cells[(t, pl)]),
               "N_h_train": sum(1 for r in frame if r["stratum"] == f"{t}|{pl}" and r["split"] == "train"),
               "N_h_test": sum(1 for r in frame if r["stratum"] == f"{t}|{pl}" and r["split"] == "test"),
               "n_h": p["n_h"],
               "n_h_train": sum(1 for r in sample if r["stratum"] == f"{t}|{pl}" and r["split"] == "train"),
               "n_h_test": sum(1 for r in sample if r["stratum"] == f"{t}|{pl}" and r["split"] == "test"),
               "f_h": p["n_h"] / len(cells[(t, pl)])}
              for t in CLASSES for pl in PLANES]
    _write(os.path.join(out, "sample_strata.csv"), strata, list(strata[0]))
    _write(os.path.join(out, "sample_size.csv"),
           [{"level": c, "N": r["N"], "n0": r["n0"], "n_exact": r["n_exact"], "n_min": r["n_min"],
             "n_drawn": p["n_h"] * len(PLANES), "margin_achieved": p["margin_class"][c]}
            for c, r in p["per_class"].items()]
           + [{"level": "all", "N": p["N"], "n0": p["overall"]["n0"], "n_exact": p["overall"]["n_exact"],
               "n_min": p["overall"]["n_min"], "n_drawn": p["n"], "margin_achieved": p["margin_overall"]}],
           ["level", "N", "n0", "n_exact", "n_min", "n_drawn", "margin_achieved"])
    print(f"Population N = {p['N']}  (per class {p['N_c']})")
    for c, r in p["per_class"].items():
        print(f"  Cochran {c:<11}: n0 = {r['n0']:.2f}, n = {r['n_exact']:.2f} -> {r['n_min']}")
    print(f"Equal allocation: n_h = ceil({p['n_class_required']}/3) = {p['n_h']} per stratum, n = {p['n']}")
    print(f"Achieved margin: overall ±{100 * p['margin_overall']:.2f}%, per class "
          + ", ".join(f"{c} ±{100 * v:.2f}%" for c, v in p["margin_class"].items()))
    return sample


def read_manifest(out=OUT):
    """[(image_path, mask_path, row)] of the fixed sample, in file order."""
    with open(os.path.join(out, "sample.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r["image"], r["mask"], r) for r in rows]


# ─── Prune ───────────────────────────────────────────────────────────────────
def prune(out=OUT, splits=SPLITS, dry_run=True):
    """Delete every image/mask in the splits that is not in the manifest."""
    keep = set()
    for ip, mp, _ in read_manifest(out):
        keep.add(os.path.normcase(os.path.abspath(ip)))
        keep.add(os.path.normcase(os.path.abspath(mp)))
    missing = [p for p in keep if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"{len(missing)} sampled files are missing on disk; refusing to prune.")
    victims = []
    for root in splits:
        for sub in ("images", "masks"):
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if os.path.isfile(p) and os.path.normcase(os.path.abspath(p)) not in keep:
                    victims.append(p)
    print(f"{len(keep)} sampled files kept; {len(victims)} files not in the sample "
          + ("would be deleted (dry run)." if dry_run else "deleted."))
    if not dry_run:
        for p in victims:
            os.remove(p)
    return victims


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--draw", action="store_true", help="draw the sample and write the manifest")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    ap.add_argument("--prune", action="store_true", help="delete files that are not in the sample")
    ap.add_argument("--yes", action="store_true", help="really delete (otherwise a dry run)")
    args = ap.parse_args()
    if args.draw:
        write_manifest(args.out, args.force)
    if args.prune:
        prune(args.out, dry_run=not args.yes)
