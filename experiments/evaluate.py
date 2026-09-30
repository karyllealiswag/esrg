"""
evaluate.py — Automated batch evaluation for Chapter 4 (Experiments E1–E6).

Purpose : Run every configuration of the study on one stratified sample of the
          BRISC 2025 test split and record the raw per-image results.
Function : draw_sample() takes a proportional stratified random sample (tumor
          class x imaging plane) of SAMPLE_SIZE slices from a split; each sampled slice is processed under
          the configurations in CONFIGS, and one row per (slice, configuration) is
          appended to the raw-results CSV. Rows already in the CSV are skipped, so
          an interrupted run resumes where it stopped. A separate sequential pass
          (--timing) re-measures processing time without parallel contention.
Notes   : Seeding conditions: "auto" uses the automated seed selection (Objective 1);
          "planted" places one click at the deepest pixel of the ground-truth tumor
          (dilated like a manual click), isolating region growing from seeding.
          "biased" multiplies a synthetic linear bias field (40% INU) into the slice
          before normalization (Experiment E2). Parameters are the frozen defaults.
          CLI: python experiments/evaluate.py --root segmentation_task/test
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one BLAS thread per worker process
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg.io_utils import list_pairs, load_image, load_mask, parse_brisc_name
from experiments.attribution import attribute

SAMPLE_FRAC = 0.80             # first-stage draw (the sampling frame of the subsample)
SAMPLE_SIZE = 500              # slices evaluated
SAMPLE_SEED = 2026
INU = 0.40                     # bias field spans 1 - INU/2 .. 1 + INU/2 (BrainWeb "40% INU")

# id: (description, config overrides, seeding condition, biased)
CONFIGS = {
    "A_SRG":           ("SRG, automatic seeding",                     dict(method="srg"), "auto", False),
    "A_ESRG":          ("ESRG, automatic seeding",                    dict(),             "auto", False),
    "P_SRG":           ("SRG, planted seed",                          dict(method="srg"), "planted", False),
    "P_ESRG":          ("ESRG (full), planted seed",                  dict(),             "planted", False),
    "P_ESRG_global":   ("ESRG, global measure, planted seed",         dict(use_log_local=False), "planted", False),
    "P_ESRG_nolog":    ("ESRG, log transform off, planted seed",      dict(use_log=False), "planted", False),
    "P_ESRG_nostop":   ("ESRG, no stopping criterion, planted seed",  dict(use_stopping=False), "planted", False),
    "P_ESRG_nopurify": ("ESRG, unpurified click disk, planted seed",  dict(purify_manual_seed=False), "planted", False),
    "B_SRG":           ("SRG, planted seed, bias field",              dict(method="srg"), "planted", True),
    "B_SRG_N4":        ("SRG + N4, planted seed, bias field",         dict(method="srg", use_n4=True), "planted", True),
    "B_ESRG":          ("ESRG (full), planted seed, bias field",      dict(), "planted", True),
    "B_ESRG_global":   ("ESRG, global measure, planted seed, bias field", dict(use_log_local=False), "planted", True),
    "B_ESRG_nolog":    ("ESRG, log transform off, planted seed, bias field", dict(use_log=False), "planted", True),
}

FIELDS = ["file", "tumor", "plane", "index", "config", "method", "seeding", "biased",
          "status", "dsc", "iou", "precision", "recall", "hd95", "assd",
          "tp", "fp", "fn", "pred_area", "gt_area", "area_ratio", "leaked", "success",
          "seed_hit", "seed_area", "seed_in_gt_frac", "planted_row", "planted_col",
          "sigma_floor", "sigma_A_initial", "k_local", "stop_reason", "n_passes",
          "bias_theta_deg", "bucket", "bucket_reason", "seconds", "error"]


# ─── Sampling ────────────────────────────────────────────────────────────────
def _allocate(sizes, n):
    """Proportional allocation n_h = n * N_h / N, rounded by the largest-remainder method."""
    N = sum(sizes.values())
    quota = {h: n * v / N for h, v in sizes.items()}
    alloc = {h: math.floor(q) for h, q in quota.items()}
    for h in sorted(quota, key=lambda h: quota[h] - alloc[h], reverse=True)[:n - sum(alloc.values())]:
        alloc[h] += 1
    return alloc


def draw_sample(root, n=SAMPLE_SIZE, frac=SAMPLE_FRAC, seed=SAMPLE_SEED):
    """
    Proportional stratified random sample of n slices. Each tumor x plane stratum h
    of size N_h receives n_h = n * N_h / N slices (largest-remainder rounding). The
    n_h slices are drawn at random, with a fixed seed, from a first-stage stratified
    draw of frac * N_h slices, so every slice of a stratum has the same chance of
    selection. Returns (sample, strata) where strata maps (tumor, plane) -> (N_h, n_h).
    """
    cells = {}
    for ip, mp in list_pairs(root):
        if mp:
            k = parse_brisc_name(ip)
            cells.setdefault((k["tumor"], k["plane"]), []).append((ip, mp))
    sizes = {h: len(v) for h, v in cells.items()}
    first = _allocate(sizes, round(frac * sum(sizes.values())))
    final = _allocate(sizes, n)
    rng, rng2 = random.Random(seed), random.Random(seed + 1)
    sample = []
    for h in sorted(cells):
        frame = rng.sample(sorted(cells[h]), first[h])
        sample += rng2.sample(sorted(frame), final[h])
    return sorted(sample), {h: (sizes[h], final[h]) for h in sorted(cells)}


# ─── Seeds and bias field ────────────────────────────────────────────────────
def planted_core(gt, cfg):
    """One click at the deepest ground-truth pixel, dilated exactly like a manual click."""
    dt = ndi.distance_transform_edt(gt)
    r, c = np.unravel_index(int(np.argmax(dt)), dt.shape)
    core = np.zeros(gt.shape, bool)
    core[r, c] = True
    return morphology.dilation(core, morphology.disk(cfg.manual_seed_radius)), (int(r), int(c))


def bias_hook(index):
    """Linear multiplicative field from 1 - INU/2 to 1 + INU/2 in a per-slice random direction."""
    theta = random.Random(10_000 + int(index)).uniform(0, 2 * math.pi)

    def hook(raw):
        H, W = raw.shape
        y, x = np.mgrid[0:H, 0:W]
        u = (x - (W - 1) / 2) * math.cos(theta) + (y - (H - 1) / 2) * math.sin(theta)
        span = np.abs(u).max() or 1.0
        return raw * (1.0 + (INU / 2) * u / span)
    return hook, math.degrees(theta)


# ─── One slice, all configurations ───────────────────────────────────────────
def process(task):
    ip, mp, cfg_ids = task
    base = Config()
    info = parse_brisc_name(ip)
    img, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, img.shape)
    pcore, (pr, pc) = planted_core(gt, base)
    rows = []
    for cid in cfg_ids:
        desc, over, seeding, biased = CONFIGS[cid]
        cfg = base.replace(**over)
        row = {"file": os.path.basename(ip), "tumor": info["tumor"], "plane": info["plane"],
               "index": info["index"], "config": cid, "method": cfg.method,
               "seeding": seeding, "biased": biased}
        try:
            hook, theta = bias_hook(info["index"]) if biased else (None, None)
            res = run(ip, cfg, mask_path=mp,
                      planted_core=pcore if seeding == "planted" else None, raw_hook=hook)
            sc = res.scores
            seed = res.stage("seed").image.astype(bool)
            row.update({k: sc.get(k) for k in ("dsc", "iou", "precision", "recall", "hd95", "assd",
                                                "tp", "fp", "fn", "pred_area", "gt_area",
                                                "area_ratio", "leaked", "seed_hit", "seconds")})
            row["status"] = res.status
            row["success"] = bool(sc.get("dsc") is not None and sc["dsc"] >= 0.70)
            row["seed_area"] = int(seed.sum())
            row["seed_in_gt_frac"] = float((seed & gt).sum() / seed.sum()) if seed.any() else None
            if seeding == "planted":
                row["planted_row"], row["planted_col"] = pr, pc
            row["sigma_floor"] = res.stage("log").info.get("noise floor σ")
            g = res.stage("growth")
            if g is not None:
                row["stop_reason"] = g.info.get("stop reason")
                passes = g.info.get("passes")
                if isinstance(passes, list) and passes and passes[0].get("sigma") is not None:
                    row["sigma_A_initial"] = passes[0]["sigma"]
                    row["n_passes"] = len(passes)
                    row["k_local"] = cfg.k_local
            if biased:
                row["bias_theta_deg"] = round(theta, 2)
            if cid == "A_ESRG":
                row["bucket"], row["bucket_reason"] = attribute(res)
        except Exception as e:                       # recorded, never silently dropped
            row["status"], row["error"] = "ERROR", repr(e)
        rows.append(row)
    return rows


def _done_keys(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as f:
        return {(r["file"], r["config"]) for r in csv.DictReader(f)}


def run_all(sample, out_csv, cfg_ids, workers):
    done = _done_keys(out_csv)
    tasks = []
    for ip, mp in sample:
        todo = [c for c in cfg_ids if (os.path.basename(ip), c) not in done]
        if todo:
            tasks.append((ip, mp, todo))
    print(f"{len(sample)} slices, {len(cfg_ids)} configurations; {len(tasks)} slices still to run", flush=True)
    new = not os.path.isfile(out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    t0 = time.time()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        with Pool(workers) as pool:
            for n, rows in enumerate(pool.imap_unordered(process, tasks), 1):
                w.writerows(rows)
                f.flush()
                if n % 25 == 0 or n == len(tasks):
                    el = time.time() - t0
                    print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                          f"~{el/n*(len(tasks)-n)/60:.1f} min left", flush=True)


def timing_pass(sample, out_csv, cfg_ids):
    """Sequential re-run (one process, nothing else running) that records only seconds."""
    done = _done_keys(out_csv)
    new = not os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "tumor", "plane", "config", "seconds"])
        if new:
            w.writeheader()
        for n, (ip, mp) in enumerate(sample, 1):
            info = parse_brisc_name(ip)
            for cid in cfg_ids:
                if (os.path.basename(ip), cid) in done:
                    continue
                _, over, seeding, _ = CONFIGS[cid]
                res = run(ip, Config().replace(**over), mask_path=mp)
                w.writerow({"file": os.path.basename(ip), "tumor": info["tumor"],
                            "plane": info["plane"], "config": cid,
                            "seconds": res.scores.get("seconds")})
            f.flush()
            if n % 50 == 0:
                print(f"  timing {n}/{len(sample)}", flush=True)


def write_sample(sample, strata, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "tumor", "plane", "index", "mask"])
        for ip, mp in sample:
            k = parse_brisc_name(ip)
            w.writerow([os.path.basename(ip), k["tumor"], k["plane"], k["index"], os.path.basename(mp)])
    with open(path.replace(".csv", "_strata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tumor", "plane", "N_h", "n_h"])
        for (t, p), (N, n) in strata.items():
            w.writerow([t, p, N, n])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="segmentation_task/test")
    ap.add_argument("--out", default="outputs/evaluation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--timing", action="store_true", help="sequential timing pass for A_SRG and A_ESRG")
    ap.add_argument("--limit", type=int, default=None, help="first N sampled slices only (smoke test)")
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE, help="sample size")
    args = ap.parse_args()

    sample, strata = draw_sample(args.root, n=args.n)
    os.makedirs(args.out, exist_ok=True)
    write_sample(sample, strata, os.path.join(args.out, "sample.csv"))
    if args.limit:
        sample = sample[:args.limit]
    if args.timing:
        timing_pass(sample, os.path.join(args.out, "timing_sequential.csv"), ["A_SRG", "A_ESRG"])
    else:
        run_all(sample, os.path.join(args.out, "raw_results.csv"), args.configs, args.workers)
