"""
sampling.py — Sample-size determination and the patient-level split of the BraTS 2023 pool.

Purpose : Split the BraTS 2023 glioma (GLI) and meningioma (MEN) pool by patient into a
          locked TEST set and a DEVELOPMENT set, record the split, and verify it.
Function : cochran() gives the minimum sample for a proportion with the finite population
          correction and margin() the achieved margin of error; write_brats() splits the
          pool by patient (seed 2026, 70% TEST / 30% DEV per type, Cochran minimum checked),
          copies only those slices into brats_eval/ and writes split.csv; verify_brats()
          runs the dataset checks and spot_check() draws ground-truth outlines on random
          slices; read_split() feeds the evaluation and the tuning.
Notes   : See experiments/PROTOCOL_BRATS.md. The TEST sample of the evaluation (300 per
          tumor type) is drawn later by experiments/evaluate.py --sample-per-type.
          CLI: python -P experiments/sampling.py --brats --draw     (patient-level split)
               python -P experiments/sampling.py --brats --verify   (dataset checks)
"""
import argparse
import csv
import math
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrg.io_utils import parse_brats_name

SEED = 2026                    # random seed of the draw (reproducible sample)
Z95 = 1.959964                 # standard normal quantile for 95% confidence
MARGIN = 0.05                  # tolerated margin of error of a proportion
P0 = 0.5                       # assumed proportion (most conservative)


# ─── Sample size ─────────────────────────────────────────────────────────────
def cochran(N, z=Z95, e=MARGIN, p=P0):
    """Cochran (1977): n0 = z^2 p(1-p) / e^2, then n = n0 / (1 + (n0 - 1) / N)."""
    n0 = z * z * p * (1 - p) / (e * e)
    n = n0 / (1 + (n0 - 1) / N)
    return {"N": N, "n0": n0, "n_exact": n, "n_min": math.ceil(n)}


def margin(n, N, z=Z95, p=P0):
    """Achieved margin of error of a proportion: e = z sqrt(p(1-p)/n * (N-n)/(N-1))."""
    return z * math.sqrt(p * (1 - p) / n * (N - n) / (N - 1))


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--draw", action="store_true", help="write the patient-level split into brats_eval/")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    ap.add_argument("--brats", action="store_true", help="select the BraTS 2023 split (with --draw or --verify)")
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
    ap.error("use --brats with --draw or --verify")
