"""
evaluate.py — Automated batch evaluation for Chapter 4 (Experiments E1–E6).

Purpose : Run every configuration of the study on the fixed stratified sample
          (experiments/sampling.py) and record one raw row per (slice, configuration).
Function : process() runs all configurations of CONFIGS on one slice and returns
          the per-slice measurements (confusion counts, overlap and boundary
          metrics, seed measurements, stage times); run_all() distributes the slices
          over worker processes and appends rows to raw_results.csv, skipping rows
          already present so an interrupted run resumes; timing_pass() re-measures
          processing time sequentially (one process, warm-up run discarded, the
          order of SRG and ESRG alternated per slice) for Experiment E6, on a
          stratified random subsample: the first TIMING_PER_STRATUM slices of every
          stratum in the random draw order of sampling.py.
Notes   : Seeding conditions —
            auto     automated seed selection (Objective 1);
            planted  one click at the deepest ground-truth pixel, dilated like a
                     manual click, isolating region growing from seeding;
            operator five simulated operator clicks per slice, each at a random
                     ground-truth pixel deep enough for the click disk to lie inside
                     the tumor (Objective 1, operator variability).
            manual   the GUI's Manual Seeding protocol for SRG: operator click k as
                     tumor seed (type 1) plus competing seeds in brain (type 2),
                     scalp/skull (type 3) and air (type 4), grown by the original
                     whole-image tessellation (growing.grow_srg). Seed positions are
                     written to manual_seeds.csv so each run can be repeated in the GUI.
          "biased" multiplies a synthetic linear bias field (40% INU) into the slice
          before normalization (Experiment E2). Parameters are the frozen defaults.
          CLI: python experiments/evaluate.py            (parallel accuracy run)
               python experiments/evaluate.py --timing   (sequential timing pass)
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
import warnings
import zlib
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one BLAS thread per worker process
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import preprocessing as pre
from esrg.io_utils import load_image, load_mask
from experiments.attribution import attribute
from experiments.sampling import read_manifest

OUT = "outputs/evaluation"
INU = 0.40                     # bias field spans 1 - INU/2 .. 1 + INU/2 (BrainWeb "40% INU")
N_CLICKS = 5                   # simulated operators per slice
SUCCESS_DSC = 0.70             # success threshold (Zijdenbos et al., 1994)
TIMING_PER_STRATUM = 20        # slices per stratum in the sequential timing pass (E6)
MANUAL_GT_CLEARANCE = 12       # px: brain/scalp seeds of the manual protocol stay this far from the tumor
MANUAL_MIN_SEP = 25            # px: minimum distance between any two manual-protocol seeds

# id: (description, config overrides, seeding condition, biased, click number)
CONFIGS = {
    "A_SRG":           ("SRG, automatic seeding",                     dict(method="srg"), "auto", False, None),
    "A_ESRG":          ("ESRG, automatic seeding",                    dict(),             "auto", False, None),
    "P_SRG":           ("SRG, planted seed",                          dict(method="srg"), "planted", False, None),
    "P_ESRG":          ("ESRG (full), planted seed",                  dict(),             "planted", False, None),
    "P_ESRG_global":   ("ESRG, global measure, planted seed",         dict(use_log_local=False), "planted", False, None),
    "P_ESRG_nolog":    ("ESRG, log transform off, planted seed",      dict(use_log=False), "planted", False, None),
    "P_ESRG_nostop":   ("ESRG, no stopping criterion, planted seed",  dict(use_stopping=False), "planted", False, None),
    "P_ESRG_nopurify": ("ESRG, unpurified click disk, planted seed",  dict(purify_manual_seed=False), "planted", False, None),
    "B_SRG":           ("SRG, planted seed, bias field",              dict(method="srg"), "planted", True, None),
    "B_SRG_N4":        ("SRG + N4, planted seed, bias field",         dict(method="srg", use_n4=True), "planted", True, None),
    "B_ESRG":          ("ESRG (full), planted seed, bias field",      dict(), "planted", True, None),
    "B_ESRG_global":   ("ESRG, global measure, planted seed, bias field", dict(use_log_local=False), "planted", True, None),
    "B_ESRG_nolog":    ("ESRG, log transform off, planted seed, bias field", dict(use_log=False), "planted", True, None),
}
for _k in range(1, N_CLICKS + 1):
    CONFIGS[f"O_SRG_{_k}"] = (f"SRG, simulated operator click {_k}", dict(method="srg"), "operator", False, _k)
for _k in range(1, N_CLICKS + 1):
    CONFIGS[f"O_ESRG_{_k}"] = (f"ESRG, simulated operator click {_k}", dict(), "operator", False, _k)
N_MANUAL = 3                   # simulated operators for the manual SRG protocol
for _k in range(1, N_MANUAL + 1):
    CONFIGS[f"M_SRG_{_k}"] = (f"SRG, manual protocol (click {_k} + competing seeds)",
                              dict(method="srg", seed_mode="manual"), "manual", False, _k)
# Manual SRG and ESRG from operator click 1 under the synthetic bias field (Objective 2)
CONFIGS["BM_SRG"] = ("SRG, manual protocol click 1, bias field",
                     dict(method="srg", seed_mode="manual"), "manual", True, 1)
CONFIGS["BM_ESRG"] = ("ESRG, simulated operator click 1, bias field", dict(), "operator", True, 1)

STAGE_KEYS = ["input", "mask", "log", "candidates", "growth", "final"]
FIELDS = ["file", "split", "tumor", "plane", "stratum", "index", "config", "method", "seeding",
          "biased", "click", "status",
          "tp", "fp", "fn", "pred_area", "gt_area",
          "dsc", "iou", "precision", "recall", "hd95", "assd", "hd95_wc", "assd_wc", "diag",
          "n_boundary_pred", "n_boundary_gt", "area_ratio", "leaked", "success",
          "seed_area", "seed_in_gt_px", "seed_in_gt_frac", "seed_hit", "seed_distance",
          "click_row", "click_col", "click_depth",
          "sigma_floor", "sigma_A_initial", "k_local", "stop_reason", "n_passes",
          "bias_theta_deg", "bucket", "bucket_reason", "seconds"] \
         + [f"t_{k}" for k in STAGE_KEYS] + ["error"]
SCORE_KEYS = ("tp", "fp", "fn", "pred_area", "gt_area", "dsc", "iou", "precision", "recall",
              "hd95", "assd", "hd95_wc", "assd_wc", "diag", "n_boundary_pred", "n_boundary_gt",
              "area_ratio", "leaked", "seed_hit", "seed_distance", "seconds")


# ─── Seeds and bias field ────────────────────────────────────────────────────
def _disk(shape, r, c, radius):
    core = np.zeros(shape, bool)
    core[r, c] = True
    return morphology.dilation(core, morphology.disk(radius))


def planted_core(gt, cfg):
    """One click at the deepest ground-truth pixel, dilated exactly like a manual click."""
    dt = ndi.distance_transform_edt(gt)
    r, c = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return _disk(gt.shape, r, c, cfg.manual_seed_radius), (int(r), int(c), float(dt[r, c]))


def operator_clicks(gt, cfg, name, k=N_CLICKS):
    """
    k simulated operator clicks: distinct ground-truth pixels drawn uniformly (fixed
    per-slice seed) among those whose distance to the tumor boundary exceeds the
    click radius, so each dilated click lies inside the tumor as a careful operator
    would place it. Tumors too thin for that fall back to any tumor pixel.
    """
    dt = ndi.distance_transform_edt(gt)
    pool = np.argwhere(dt > cfg.manual_seed_radius)
    if len(pool) < k:
        pool = np.argwhere(gt)
    rng = random.Random(zlib.crc32(name.encode()))
    picks = rng.sample(range(len(pool)), min(k, len(pool)))
    return [(int(pool[i][0]), int(pool[i][1]), float(dt[pool[i][0], pool[i][1]])) for i in picks]


def _pick(pool, n, rng, taken, min_sep):
    """Up to n pixels from pool ((N, 2) array), each at least min_sep from the others and from taken."""
    out = []
    order = list(range(len(pool)))
    rng.shuffle(order)
    for i in order:
        r, c = int(pool[i][0]), int(pool[i][1])
        if all((r - tr) ** 2 + (c - tc) ** 2 >= min_sep ** 2 for tr, tc in taken + out):
            out.append((r, c))
            if len(out) == n:
                break
    return out


def manual_protocol_points(raw, gt, cfg, name, k, click):
    """
    Seeds of a simulated operator using the GUI's Manual Seeding for SRG: the tumor click
    (type 1) plus competing seeds in brain tissue (type 2, two), scalp/skull (type 3,
    three) and air (type 4, three), as a careful operator would place them. Positions are
    deterministic per (slice, operator); brain and scalp seeds stay clear of the tumor.
    Returns [(row, col, type), ...] for run(manual_points=...).
    """
    img = pre.normalize(raw)
    mask, depth, _ = pre.head_mask(img, cfg)
    rng = random.Random(zlib.crc32(f"{name}M{k}".encode()))
    far = ndi.distance_transform_edt(~gt) > MANUAL_GT_CLEARANCE
    brain = np.argwhere(mask & far & (depth > 0.4 * depth[mask].max()))
    scalp = np.argwhere(mask & far & (depth >= 2) & (depth <= 8))
    air = np.argwhere(~mask & (ndi.distance_transform_edt(~mask) >= 8))
    pts, taken = [(click[0], click[1], 1)], [tuple(click)]
    for t, pool, n in ((2, brain, 2), (3, scalp, 3), (4, air, 3)):
        got = _pick(pool, n, rng, taken, MANUAL_MIN_SEP)
        pts += [(r, c, t) for r, c in got]
        taken += got
    return pts


def bias_hook(index, split):
    """Linear multiplicative field from 1 - INU/2 to 1 + INU/2 in a per-slice random direction."""
    theta = random.Random(zlib.crc32(f"{split}{index}".encode())).uniform(0, 2 * math.pi)

    def hook(raw):
        H, W = raw.shape
        y, x = np.mgrid[0:H, 0:W]
        u = (x - (W - 1) / 2) * math.cos(theta) + (y - (H - 1) / 2) * math.sin(theta)
        span = np.abs(u).max() or 1.0
        return raw * (1.0 + (INU / 2) * u / span)
    return hook, math.degrees(theta)


# ─── One slice, all configurations ───────────────────────────────────────────
def _row_from_result(row, res, gt, cfg):
    sc = res.scores
    row.update({k: sc.get(k) for k in SCORE_KEYS})
    row["status"] = res.status
    row["success"] = bool(sc.get("dsc") is not None and sc["dsc"] >= SUCCESS_DSC)
    seed = res.stage("seed").image.astype(bool)
    row["seed_area"] = int(seed.sum())
    row["seed_in_gt_px"] = int((seed & gt).sum())
    row["seed_in_gt_frac"] = float((seed & gt).sum() / seed.sum()) if seed.any() else None
    row["sigma_floor"] = res.stage("log").info.get("noise floor σ") if res.stage("log") else None
    g = res.stage("growth")
    if g is not None:
        row["stop_reason"] = g.info.get("stop reason")
        passes = g.info.get("passes")
        if isinstance(passes, list) and passes and passes[0].get("sigma") is not None:
            row["sigma_A_initial"] = passes[0]["sigma"]
            row["n_passes"] = len(passes)
            row["k_local"] = cfg.k_local
    for k in STAGE_KEYS:
        st = res.stage(k)
        row[f"t_{k}"] = round(st.seconds, 5) if st is not None else None
    return row


def process(task):
    ip, mp, info, cfg_ids = task
    base = Config()
    img, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, img.shape)
    pcore, (pr, pc, pd) = planted_core(gt, base)
    clicks = operator_clicks(gt, base, info["file"])
    rows = []
    for cid in cfg_ids:
        desc, over, seeding, biased, click = CONFIGS[cid]
        cfg = base.replace(**over)
        row = {"file": info["file"], "split": info["split"], "tumor": info["tumor"],
               "plane": info["plane"], "stratum": info["stratum"], "index": info["index"],
               "config": cid, "method": cfg.method, "seeding": seeding, "biased": biased,
               "click": click}
        try:
            hook, theta = bias_hook(info["index"], info["split"]) if biased else (None, None)
            core, points = None, None
            if seeding == "planted":
                core = pcore
                row["click_row"], row["click_col"], row["click_depth"] = pr, pc, round(pd, 3)
            elif seeding == "operator":
                cr, cc, cd = clicks[(click - 1) % len(clicks)]
                core = _disk(gt.shape, cr, cc, base.manual_seed_radius)
                row["click_row"], row["click_col"], row["click_depth"] = cr, cc, round(cd, 3)
            elif seeding == "manual":
                cr, cc, cd = clicks[(click - 1) % len(clicks)]
                points = manual_protocol_points(img, gt, base, info["file"], click, (cr, cc))
                row["click_row"], row["click_col"], row["click_depth"] = cr, cc, round(cd, 3)
                row["manual_points"] = json.dumps(points)     # -> manual_seeds.csv, not raw_results.csv
            res = run(ip, cfg, mask_path=mp, planted_core=core, manual_points=points, raw_hook=hook)
            _row_from_result(row, res, gt, cfg)
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
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["file"], r["config"]) for r in csv.DictReader(f)}


def _tasks(sample, cfg_ids, done):
    tasks = []
    for ip, mp, info in sample:
        todo = [c for c in cfg_ids if (info["file"], c) not in done]
        if todo:
            tasks.append((ip, mp, info, todo))
    return tasks


def run_all(sample, out_csv, cfg_ids, workers):
    tasks = _tasks(sample, cfg_ids, _done_keys(out_csv))
    print(f"{len(sample)} slices, {len(cfg_ids)} configurations; {len(tasks)} slices still to run", flush=True)
    new = not os.path.isfile(out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    t0 = time.time()
    seeds_csv = os.path.join(os.path.dirname(out_csv) or ".", "manual_seeds.csv")
    new_seeds = not os.path.isfile(seeds_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f, \
            open(seeds_csv, "a", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        sw = csv.writer(fs)
        if new:
            w.writeheader()
        if new_seeds:
            sw.writerow(["file", "config", "points"])      # points: JSON [[row, col, type], ...]
        with Pool(workers) as pool:
            for n, rows in enumerate(pool.imap_unordered(process, tasks), 1):
                w.writerows(rows)
                sw.writerows([r["file"], r["config"], r["manual_points"]] for r in rows if r.get("manual_points"))
                f.flush()
                fs.flush()
                if n % 10 == 0 or n == len(tasks):
                    el = time.time() - t0
                    print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                          f"~{el/n*(len(tasks)-n)/60:.1f} min left", flush=True)


# ─── Sequential timing pass (E6) ─────────────────────────────────────────────
TIMING_FIELDS = ["file", "split", "tumor", "plane", "stratum", "config", "order", "seconds"] \
                + [f"t_{k}" for k in STAGE_KEYS] + ["status", "dsc", "pred_area"]


def environment():
    """Hardware/software of the timing pass (reported with Table 3.1)."""
    import subprocess
    import numpy, scipy, skimage
    cpu = platform.processor()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name; "
                              "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"],
                             capture_output=True, text=True, timeout=30).stdout.split("\n")
        cpu, ram = out[0].strip() or cpu, out[1].strip()
    except Exception:
        ram = None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Battery).BatteryStatus; "
                              "(powercfg /getactivescheme)"], capture_output=True, text=True,
                             timeout=30).stdout.splitlines()
        power = {"battery_status": out[0].strip(), "power_scheme": out[1].strip()}
        power["source"] = ("battery" if power["battery_status"] == "1" else
                           "AC adapter" if power["battery_status"] else "AC (no battery)")
    except Exception:
        power = {}
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": ram, **power,
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python": platform.python_version(), "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit_image": skimage.__version__,
            "timer": "time.perf_counter", "processes": 1}


def timing_subsample(sample, k=TIMING_PER_STRATUM):
    """First k slices of every stratum in the random draw order: a stratified random subsample."""
    rows = sorted(sample, key=lambda t: (t[2]["stratum"], int(t[2]["draw_order"])))
    out, seen = [], {}
    for t in rows:
        h = t[2]["stratum"]
        if seen.get(h, 0) < k:
            out.append(t)
            seen[h] = seen.get(h, 0) + 1
    return sorted(out, key=lambda t: t[2]["file"])


def timing_pass(sample, out_csv, cfg_ids=("A_SRG", "A_ESRG")):
    """
    One process, nothing else running. A warm-up run (discarded) loads libraries
    and caches; the order of the two configurations alternates between slices so
    neither systematically benefits from a warm cache.
    """
    done = _done_keys(out_csv)
    new = not os.path.isfile(out_csv)
    with open(os.path.join(os.path.dirname(out_csv), "timing_environment.json"), "w") as f:
        json.dump(environment(), f, indent=2)
    ip0, mp0, _ = sample[0]
    for cid in cfg_ids:                              # warm-up, not recorded
        run(ip0, Config().replace(**CONFIGS[cid][1]), mask_path=mp0)
    t0 = time.time()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TIMING_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for n, (ip, mp, info) in enumerate(sample, 1):
            order = list(cfg_ids) if n % 2 else list(reversed(cfg_ids))
            for pos, cid in enumerate(order, 1):
                if (info["file"], cid) in done:
                    continue
                res = run(ip, Config().replace(**CONFIGS[cid][1]), mask_path=mp)
                row = {k: info[k] for k in ("file", "split", "tumor", "plane", "stratum")}
                row.update({"config": cid, "order": pos, "seconds": res.meta.get("seconds"),
                            "status": res.status, "dsc": res.scores.get("dsc"),
                            "pred_area": res.scores.get("pred_area")})
                for k in STAGE_KEYS:
                    st = res.stage(k)
                    row[f"t_{k}"] = round(st.seconds, 5) if st is not None else None
                w.writerow(row)
            f.flush()
            if n % 25 == 0 or n == len(sample):
                el = time.time() - t0
                print(f"  timing {n}/{len(sample)}  {el/60:.1f} min elapsed, "
                      f"~{el/n*(len(sample)-n)/60:.1f} min left", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--timing", action="store_true", help="sequential timing pass for A_SRG and A_ESRG")
    ap.add_argument("--limit", type=int, default=None, help="first N sampled slices only (smoke test)")
    ap.add_argument("--raw", default="raw_results.csv", help="raw-results file name inside --out")
    ap.add_argument("--per-stratum", type=int, default=TIMING_PER_STRATUM,
                    help="slices per stratum timed in the sequential pass")
    args = ap.parse_args()

    sample = read_manifest(args.out)
    if args.limit:
        sample = sample[:args.limit]
    if args.timing:
        sub = timing_subsample(sample, args.per_stratum)
        print(f"timing pass on {len(sub)} slices ({args.per_stratum} per stratum)", flush=True)
        timing_pass(sub, os.path.join(args.out, "timing_sequential.csv"))
    else:
        run_all(sample, os.path.join(args.out, args.raw), args.configs, args.workers)
