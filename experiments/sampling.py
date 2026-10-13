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
          BraTS 2023 (experiments/PROTOCOL_BRATS.md): write_brats() splits the GLI and
          MEN pool by patient (seed 2026, 70% TEST / 30% DEV per type, Cochran minimum
          checked), copies only those slices into brats_eval/ and writes split.csv;
          verify_brats() runs the dataset checks; read_split() feeds evaluation.
          CLI: python experiments/sampling.py --draw            (write manifest)
               python experiments/sampling.py --prune --yes     (delete unused files)
               python -P experiments/sampling.py --brats --draw     (BraTS split)
               python -P experiments/sampling.py --brats --verify   (BraTS checks)
"""
import argparse
import csv
import math
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrg.io_utils import list_pairs, parse_brisc_name, parse_brats_name

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


# ─── BraTS 2023 (GLI + MEN): patient-level split (experiments/PROTOCOL_BRATS.md) ──
BRATS_POOL = os.path.join(os.path.expanduser("~"), "Downloads", "brats_pool", "brats_pool")
BRATS_OUT = "brats_eval"
BRATS_TYPES = ["glioma", "meningioma"]
TEST_FRAC = 0.70               # share of each type's patients in the locked TEST set
SPLIT_FIELDS = ["patient", "tumor", "set", "session", "slice", "file", "core_area",
                "enhancing_area", "draw_order"]


def brats_population(pool=BRATS_POOL):
    """
    {tumor: [manifest row, ...]} of the pool, one scan per patient. A patient is the
    5-digit ID; with repeat scans the lowest session is kept (none occur in the pool,
    but the rule is fixed in advance). Each pool scan holds exactly one eligible slice.
    """
    pop = {}
    for t in BRATS_TYPES:
        with open(os.path.join(pool, t, "manifest.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        by_patient = {}
        for r in rows:
            k = parse_brats_name(r["file"])
            if k["tumor"] != t:
                raise SystemExit(f"{r['file']}: name does not match tumor type {t}")
            if int(r["core_area"]) < 150:
                raise SystemExit(f"{r['file']}: core area {r['core_area']} < 150, not eligible")
            r = {**r, "tumor": t, "patient_id": k["patient"], "session": k["session"], "slice": k["slice"]}
            by_patient.setdefault(k["patient"], []).append(r)
        chosen = []
        for pid, scans in by_patient.items():
            first = min(s["session"] for s in scans)
            same = [s for s in scans if s["session"] == first]
            if len(same) != 1:
                raise SystemExit(f"patient {pid}: {len(same)} slices in session {first}; expected 1")
            chosen.append(same[0])
        pop[t] = sorted(chosen, key=lambda r: r["patient_id"])
    return pop


def brats_plan(pop, test_frac=TEST_FRAC):
    """Cochran minimum per type (reported) and the 70/30 allocation decided at Checkpoint 1."""
    out = {}
    for t, rows in pop.items():
        N = len(rows)
        n_test = round(test_frac * N)
        out[t] = {"N": N, "cochran": cochran(N), "n_test": n_test, "n_dev": N - n_test,
                  "margin_fpc": margin(n_test, N), "margin_inf": Z95 * math.sqrt(P0 * (1 - P0) / n_test)}
        assert n_test >= out[t]["cochran"]["n_min"], f"{t}: TEST below the Cochran minimum"
    return out


def brats_draw(pop, plan_, seed=SEED):
    """Per type: patients sorted by ID, shuffled with a fixed seed; the first n_test go to TEST."""
    rows = []
    for t in BRATS_TYPES:
        rng = random.Random(seed)
        items = pop[t]
        order = list(range(len(items)))
        rng.shuffle(order)
        for pos, i in enumerate(order):
            r = items[i]
            rows.append({"patient": r["patient_id"], "tumor": t,
                         "set": "test" if pos < plan_[t]["n_test"] else "dev",
                         "session": r["session"], "slice": r["slice"], "file": r["file"],
                         "core_area": int(r["core_area"]), "enhancing_area": int(r["enhancing_area"]),
                         "draw_order": pos + 1})
    return sorted(rows, key=lambda r: (r["set"], r["tumor"], r["patient"]))


def write_brats(pool=BRATS_POOL, out=BRATS_OUT, force=False, seed=SEED):
    """
    Copy only the chosen slices into out/{test,dev}/{images,masks,masks_enhancing}
    (byte-for-byte; the pool is never modified) and write out/split.csv plus the
    sample-size record out/sample_size.csv.
    """
    split_csv = os.path.join(out, "split.csv")
    if os.path.isfile(split_csv) and not force:
        raise SystemExit(f"{split_csv} exists; the split is fixed. Use --force to redraw.")
    pop = brats_population(pool)
    p = brats_plan(pop)
    rows = brats_draw(pop, p, seed)
    for r in rows:
        for sub in ("images", "masks", "masks_enhancing"):
            d = os.path.join(out, r["set"], sub)
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(os.path.join(pool, r["tumor"], sub, r["file"]), os.path.join(d, r["file"]))
    _write(split_csv, rows, SPLIT_FIELDS)
    _write(os.path.join(out, "sample_size.csv"),
           [{"tumor": t, "N": v["N"], "n0": v["cochran"]["n0"], "cochran_n": v["cochran"]["n_exact"],
             "cochran_min": v["cochran"]["n_min"], "n_test": v["n_test"], "n_dev": v["n_dev"],
             "margin_fpc": v["margin_fpc"], "margin_no_fpc": v["margin_inf"]} for t, v in p.items()],
           ["tumor", "N", "n0", "cochran_n", "cochran_min", "n_test", "n_dev", "margin_fpc", "margin_no_fpc"])
    for t, v in p.items():
        c = v["cochran"]
        print(f"{t:<10} N = {v['N']}: Cochran n0 = {c['n0']:.2f}, n = {c['n_exact']:.2f} -> {c['n_min']}; "
              f"TEST = round({TEST_FRAC} x {v['N']}) = {v['n_test']}, DEV = {v['n_dev']}; "
              f"margin ±{100 * v['margin_fpc']:.1f}% (FPC), ±{100 * v['margin_inf']:.1f}% (no FPC)")
    return rows


def read_split(out=BRATS_OUT, which="test"):
    """[(image_path, mask_path, row)] of one set ('test' or 'dev') from split.csv, in file order."""
    with open(os.path.join(out, "split.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] == which]
    return [(os.path.join(out, which, "images", r["file"]).replace("\\", "/"),
             os.path.join(out, which, "masks", r["file"]).replace("\\", "/"), r)
            for r in sorted(rows, key=lambda r: r["file"])]


def verify_brats(out=BRATS_OUT, pool=BRATS_POOL):
    """Every Stage 2 check of the protocol; returns (checks, failures)."""
    import numpy as np
    from PIL import Image
    with open(os.path.join(out, "split.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    checks, fails = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if not ok:
            fails.append(name)

    sets = {s: [r for r in rows if r["set"] == s] for s in ("test", "dev")}
    for s, rs in sets.items():
        files = {r["file"] for r in rs}
        for sub in ("images", "masks", "masks_enhancing"):
            on_disk = set(os.listdir(os.path.join(out, s, sub)))
            check(f"{s}/{sub} matches split.csv", on_disk == files,
                  f"{len(on_disk)} files, {len(files)} rows")
        # GLI and MEN number their patients independently: a patient is (tumor, ID).
        pats = [(r["tumor"], r["patient"]) for r in rs]
        check(f"{s}: one slice per patient", len(pats) == len(set(pats)), f"{len(pats)} rows, {len(set(pats))} patients")
    overlap = {(r["tumor"], r["patient"]) for r in sets["test"]} & {(r["tumor"], r["patient"]) for r in sets["dev"]}
    check("no patient in both TEST and DEV", not overlap, f"{len(overlap)} shared")
    pop = brats_population(pool)
    p = brats_plan(pop)
    for t in BRATS_TYPES:
        nt = sum(1 for r in sets["test"] if r["tumor"] == t)
        nd = sum(1 for r in sets["dev"] if r["tumor"] == t)
        check(f"{t}: TEST/DEV sizes as planned", (nt, nd) == (p[t]["n_test"], p[t]["n_dev"]),
              f"TEST {nt}, DEV {nd} of N = {p[t]['N']}")
        check(f"{t}: every pool patient used once", nt + nd == p[t]["N"])
    for r in rows:
        k = parse_brats_name(r["file"])
        if k["tumor"] != r["tumor"] or k["patient"] != r["patient"]:
            fails.append(f"name/split mismatch {r['file']}")
    check("file names carry the tumor type and patient of split.csv", not any(f.startswith("name/") for f in fails))
    bad_vals, empty, size, area_mismatch = [], [], set(), 0
    for r in rows:
        im = Image.open(os.path.join(out, r["set"], "images", r["file"]))
        m = np.asarray(Image.open(os.path.join(out, r["set"], "masks", r["file"])))
        size.add((im.size, m.shape))
        v = set(np.unique(m).tolist())
        if not v <= {0, 255}:
            bad_vals.append(r["file"])
        if not (m > 127).any():
            empty.append(r["file"])
    check("mask values are {0, 255}", not bad_vals, f"{len(bad_vals)} bad")
    check("masks are non-empty", not empty, f"{len(empty)} empty")
    check("image and mask sizes agree (512 x 512)", size == {((512, 512), (512, 512))}, str(size))
    for s in ("test", "dev"):
        for ip, mp, _ in read_split(out, s)[:50]:
            from esrg.io_utils import find_mask_for
            if os.path.normcase(os.path.abspath(find_mask_for(ip) or "")) != os.path.normcase(os.path.abspath(mp)):
                fails.append(f"find_mask_for {ip}")
    check("esrg.io_utils.find_mask_for pairs images with masks", not any(f.startswith("find_mask_for") for f in fails))
    return checks, fails


def spot_check(out=BRATS_OUT, dest="outputs/brats_spotcheck", per_type=3, which="test", seed=SEED):
    """Ground-truth outline (green) drawn on per_type random slices of each type."""
    import numpy as np
    from PIL import Image
    from esrg import visualize as viz
    os.makedirs(dest, exist_ok=True)
    rows = [r for _, _, r in read_split(out, which)]
    rng = random.Random(seed)
    paths = []
    for t in BRATS_TYPES:
        for r in rng.sample([r for r in rows if r["tumor"] == t], per_type):
            img = np.asarray(Image.open(os.path.join(out, which, "images", r["file"])).convert("L"))
            gt = np.asarray(Image.open(os.path.join(out, which, "masks", r["file"]))) > 127
            rgb = np.stack([img] * 3, -1).copy()
            rgb[viz.outline(gt, 2)] = viz.GREEN
            p = os.path.join(dest, f"{which}_{r['file']}")
            Image.fromarray(rgb).save(p)
            paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--draw", action="store_true", help="draw the sample and write the manifest")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    ap.add_argument("--prune", action="store_true", help="delete files that are not in the sample")
    ap.add_argument("--yes", action="store_true", help="really delete (otherwise a dry run)")
    ap.add_argument("--brats", action="store_true", help="BraTS 2023 split into brats_eval/ (with --draw)")
    ap.add_argument("--pool", default=BRATS_POOL, help="BraTS pool folder (glioma/, meningioma/)")
    ap.add_argument("--verify", action="store_true", help="run the BraTS dataset checks")
    args = ap.parse_args()
    if args.brats:
        if args.draw:
            write_brats(args.pool, BRATS_OUT, args.force)
        if args.verify:
            checks, fails = verify_brats(BRATS_OUT, args.pool)
            for name, ok, detail in checks:
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
            print("spot-check overlays:")
            for p in spot_check():
                print("  " + p)
            sys.exit(1 if fails else 0)
        sys.exit(0)
    if args.draw:
        write_manifest(args.out, args.force)
    if args.prune:
        prune(args.out, dry_run=not args.yes)
