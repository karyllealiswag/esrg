"""
evaluate_fair.py — Fair, objective-isolating ESRG vs. SRG evaluation on BRISC 2025.

Purpose : Run Experiments E1–E8 of experiments/PROTOCOL_BRISC_FAIR.md on a 25% subsample
          of the original 954-slice BRISC sample, giving SRG and ESRG identical seeds and
          isolating each proposed method in its own arm.
Function : draw_sample() takes the first 27 already-selected slices of every tumor x
          plane stratum in their recorded random draw order (nested in the original
          sample, seed 2026); seed_protocol() places one tumor, healthy-tissue,
          background (air) and scalp/skull seed per placement P1-P3, deterministically;
          process() runs all arms of CONFIGS on one slice; run_all() spreads slices over
          worker processes and appends rows to the raw CSV, skipping finished
          (slice, arm) pairs so an interrupted run resumes. --repeat re-runs the
          automatic arms for the determinism check.
Notes   : Parameters are the BRISC-frozen defaults of esrg/config.py (not re-tuned).
          Seed pools follow the original BRISC manual protocol (evaluate_brisc.py,
          manual_protocol_points), reduced to one seed per type. Distances are in pixels:
          BRISC has no pixel spacing. The ground truth places only the planted seeds (a
          careful operator); the automatic tumor seed never sees it.
          CLI: python -P experiments/evaluate_fair.py --draw            (write sample_25.csv)
               python -P experiments/evaluate_fair.py --limit 3         (smoke test)
               python -P experiments/evaluate_fair.py                   (the full run)
               python -P experiments/evaluate_fair.py --repeat 2        (determinism)
"""
import argparse
import csv
import json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import preprocessing as pre
from esrg.io_utils import load_image, load_mask
from experiments.attribution import attribute

SRC = "outputs/evaluation"         # the original BRISC sample (read only)
OUT = "outputs/evaluation_fair"
PER_STRATUM = 27                   # 9 strata x 27 = 243 slices = 25.5% of the 954-slice sample
SUCCESS_DSC = 0.70                 # success threshold (Zijdenbos et al., 1994)
NEAR = (8.0, 16.0)                 # P2/P3 lie 8-16 px from P1
SEP = 8.0                          # P3 lies >= 8 px from P2
GT_CLEAR = 12                      # healthy and scalp seeds stay >= 12 px from the tumor
AIR_CLEAR = 8                      # air seeds lie >= 8 px outside the head
TYPES = {1: "tumor", 2: "healthy tissue", 3: "background (air)", 4: "scalp/skull"}

# id: (description, config overrides, seeding, placement, seed types used)
#   seeding "auto": automatic tumor seed + the planted competitors (types 2-4);
#   seeding "manual": every planted type listed.
CONFIGS = {
    "A_ESRG":       ("ESRG, automatic seeding",                           dict(),                                       "auto", 1, (2, 3, 4)),
    "A_SRG":        ("SRG, automatic tumor seed + planted competitors",   dict(method="srg"),                           "auto", 1, (2, 3, 4)),
    "P_SRG":        ("SRG (1994), planted P1",                            dict(method="srg"),                           "manual", 1, (1, 2, 3, 4)),
    "P_SRG_2":      ("SRG (1994), planted P2",                            dict(method="srg"),                           "manual", 2, (1, 2, 3, 4)),
    "P_SRG_3":      ("SRG (1994), planted P3",                            dict(method="srg"),                           "manual", 3, (1, 2, 3, 4)),
    "P_ESRG":       ("ESRG, planted P1 tumor seed",                       dict(),                                       "manual", 1, (1, 2, 3, 4)),
    "P_SRG_L":      ("SRG + LLDM, planted P1",                            dict(method="srg", srg_use_log_local=True),   "manual", 1, (1, 2, 3, 4)),
    "P_ESRG_noL":   ("ESRG without LLDM (ASC on), planted P1",            dict(use_log=False, use_log_local=False),     "manual", 1, (1, 2, 3, 4)),
    "P_SRG_S":      ("SRG + ASC, planted P1 with competitors",            dict(method="srg", srg_use_stopping=True),    "manual", 1, (1, 2, 3, 4)),
    "P_SRG_solo":   ("SRG (1994), P1 tumor seed only",                    dict(method="srg"),                           "manual", 1, (1,)),
    "P_SRG_solo_S": ("SRG + ASC, P1 tumor seed only",                     dict(method="srg", srg_use_stopping=True),    "manual", 1, (1,)),
    "P_ESRG_C":     ("ESRG compete mode (LLDM + ASC), planted P1",        dict(esrg_compete=True),                      "manual", 1, (1, 2, 3, 4)),
    "P_ESRG_C_noS": ("ESRG compete mode without ASC, planted P1",         dict(esrg_compete=True, use_stopping=False),  "manual", 1, (1, 2, 3, 4)),
}
AUTO_CONFIGS = [c for c, v in CONFIGS.items() if v[2] == "auto"]

STAGE_KEYS = ["input", "mask", "log", "candidates", "growth", "final"]
FIELDS = ["file", "split", "tumor", "plane", "stratum", "config", "method", "seeding", "placement", "status",
          "tp", "fp", "fn", "pred_area", "gt_area", "dsc", "iou", "precision", "recall", "hd95", "hd95_wc",
          "assd", "area_ratio", "leaked", "success", "seed_area", "seed_in_gt_frac", "seed_hit",
          "seed_distance", "mask_sha1", "tumor_seed_px_from_p1", "sigma_floor", "sigma_A_initial",
          "stop_reason", "n_passes", "bucket", "bucket_reason", "seconds"] \
         + [f"t_{k}" for k in STAGE_KEYS] + ["error"]
SCORE_KEYS = ("tp", "fp", "fn", "pred_area", "gt_area", "dsc", "iou", "precision", "recall",
              "hd95", "hd95_wc", "assd", "area_ratio", "leaked", "seed_hit", "seed_distance",
              "mask_sha1", "seconds")
SAMPLE_FIELDS = ["file", "split", "index", "tumor", "plane", "stratum", "draw_order", "image", "mask"]


# ─── Sample ──────────────────────────────────────────────────────────────────
def draw_sample(src=SRC, k=PER_STRATUM):
    """First k already-selected slices of each stratum in the recorded random draw order."""
    with open(os.path.join(src, "frame.csv"), newline="") as f:
        frame = [r for r in csv.DictReader(f) if r["selected"] == "True"]
    rows = []
    for h in sorted({r["stratum"] for r in frame}):
        st = sorted((r for r in frame if r["stratum"] == h), key=lambda r: int(r["draw_order"]))
        if len(st) < k:
            raise SystemExit(f"stratum {h}: only {len(st)} selected slices")
        rows += st[:k]
    return sorted(rows, key=lambda r: r["file"])


def write_sample(out=OUT, force=False):
    path = os.path.join(out, "sample_25.csv")
    if os.path.isfile(path) and not force:
        raise SystemExit(f"{path} exists; the subsample is fixed. Use --force to redraw.")
    rows = draw_sample()
    os.makedirs(out, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["stratum"], {"train": 0, "test": 0})[r["split"]] += 1
    print(f"{len(rows)} slices ({PER_STRATUM} per stratum, {100 * len(rows) / 954:.1f}% of 954)")
    for h, v in sorted(by.items()):
        print(f"  {h:<22} train {v['train']:>2}  test {v['test']:>2}")
    return rows


def read_sample(out=OUT):
    with open(os.path.join(out, "sample_25.csv"), newline="") as f:
        return [(r["image"], r["mask"], r) for r in csv.DictReader(f)]


# ─── Seed protocol ───────────────────────────────────────────────────────────
def _rng(file, *parts):
    return random.Random(zlib.crc32("|".join([file, *map(str, parts)]).encode()))


def _near(pool, anchor, rng, avoid=None):
    """
    A pool pixel ((N, 2) array) 8-16 px from anchor (and >= 8 px from avoid); if that
    ring is empty, the pool pixel whose distance to anchor is closest to 12 px.
    """
    d = np.hypot(pool[:, 0] - anchor[0], pool[:, 1] - anchor[1])
    ok = (d >= NEAR[0]) & (d <= NEAR[1])
    if avoid is not None:
        ok &= np.hypot(pool[:, 0] - avoid[0], pool[:, 1] - avoid[1]) >= SEP
    idx = np.flatnonzero(ok)
    if idx.size:
        i = idx[rng.randrange(idx.size)]
    else:
        far = d > 0
        if avoid is not None:
            far &= (pool[:, 0] != avoid[0]) | (pool[:, 1] != avoid[1])
        cand = np.flatnonzero(far)
        if not cand.size:
            return tuple(int(v) for v in anchor)
        i = cand[np.argmin(np.abs(d[cand] - 12.0))]
    return int(pool[i][0]), int(pool[i][1])


def _pick(pool, file, t):
    rng = _rng(file, "P1", t)
    return tuple(int(v) for v in pool[rng.randrange(len(pool))])


def seed_protocol(raw, gt, cfg, file):
    """
    ({placement: [(row, col, type), ...]}, log) for P1-P3: one seed of each type 1-4.
    Pools as in the original BRISC manual protocol; deterministic per file.
    """
    img = pre.normalize(raw)
    head, depth, _ = pre.head_mask(img, cfg)
    far = ndi.distance_transform_edt(~gt) >= GT_CLEAR
    log = {}

    dt = ndi.distance_transform_edt(gt)
    tum = np.argwhere(dt >= cfg.manual_seed_radius)
    if not len(tum):
        tum = np.argwhere(gt)
    p1 = tuple(int(v) for v in np.unravel_index(int(np.argmax(dt)), dt.shape))
    t2 = _near(tum, p1, _rng(file, "P2", 1))
    t3 = _near(tum, p1, _rng(file, "P3", 1), avoid=t2)
    seeds = {1: [(*p1, 1)], 2: [(*t2, 1)], 3: [(*t3, 1)]}

    pools = {2: head & far & (depth > 0.4 * depth[head].max()),
             3: ~head & (ndi.distance_transform_edt(~head) >= AIR_CLEAR),
             4: head & far & (depth >= 2) & (depth <= 8)}
    for t, m in pools.items():
        pool = np.argwhere(m)
        if not len(pool):                       # very small or cropped heads
            log[f"type{t}_missing"] = True
            continue
        s1 = _pick(pool, file, t)
        s2 = _near(pool, s1, _rng(file, "P2", t))
        s3 = _near(pool, s1, _rng(file, "P3", t), avoid=s2)
        for k, s in ((1, s1), (2, s2), (3, s3)):
            seeds[k].append((*s, t))
    log["tumor_px_from_p1"] = {k: round(float(np.hypot(s[0][0] - p1[0], s[0][1] - p1[1])), 2)
                               for k, s in seeds.items()}
    return seeds, log


# ─── One slice, all arms ─────────────────────────────────────────────────────
def _row_from_result(row, res, gt):
    sc = res.scores
    row.update({k: sc.get(k) for k in SCORE_KEYS})
    row["status"] = res.status
    row["success"] = bool(sc.get("dsc") is not None and sc["dsc"] >= SUCCESS_DSC)
    seed = res.stage("seed").image.astype(bool)
    row["seed_area"] = int(seed.sum())
    row["seed_in_gt_frac"] = float((seed & gt).sum() / seed.sum()) if seed.any() else None
    row["sigma_floor"] = res.stage("log").info.get("noise floor σ") if res.stage("log") else None
    g = res.stage("growth")
    if g is not None:
        row["stop_reason"] = g.info.get("stop reason")
        passes = g.info.get("passes")
        if isinstance(passes, list) and passes and passes[0].get("sigma") is not None:
            row["sigma_A_initial"] = passes[0]["sigma"]
            row["n_passes"] = len(passes)
    for k in STAGE_KEYS:
        st = res.stage(k)
        row[f"t_{k}"] = round(st.seconds, 5) if st is not None else None
    return row


def process(task):
    ip, mp, info, cfg_ids = task
    base = Config()
    raw, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, raw.shape)
    seeds, slog = seed_protocol(raw, gt, base, info["file"])
    rows = []
    for cid in cfg_ids:
        desc, over, seeding, placement, types = CONFIGS[cid]
        cfg = base.replace(seed_mode=seeding, **over)
        row = {k: info[k] for k in ("file", "split", "tumor", "plane", "stratum")}
        row.update({"config": cid, "method": cfg.method, "seeding": seeding, "placement": placement,
                    "tumor_seed_px_from_p1": slog["tumor_px_from_p1"][placement] if 1 in types else None})
        try:
            points = [p for p in seeds[placement] if p[2] in types]
            res = run(ip, cfg, mask_path=mp, manual_points=points)
            _row_from_result(row, res, gt)
            if cid == "A_ESRG":
                row["bucket"], row["bucket_reason"] = attribute(res)
        except Exception as e:                       # recorded, never silently dropped
            row["status"], row["error"] = "ERROR", repr(e)
        rows.append(row)
    seed_rows = [[info["file"], k, json.dumps(v), json.dumps({x: y for x, y in slog.items()
                                                              if x != "tumor_px_from_p1"})]
                 for k, v in seeds.items()]
    return rows, seed_rows


def _repair(path):
    """Cut a truncated last line (a crash mid-write) so the file parses cleanly."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def _done_keys(path):
    _repair(path)
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["file"], r["config"]) for r in csv.DictReader(f)}


def run_all(sample, out_csv, cfg_ids, workers, seeds_csv=None):
    done = _done_keys(out_csv)
    tasks = []
    for ip, mp, info in sample:
        todo = [c for c in cfg_ids if (info["file"], c) not in done]
        if todo:
            tasks.append((ip, mp, info, todo))
    print(f"{len(sample)} slices, {len(cfg_ids)} arms; {len(tasks)} slices still to run", flush=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    new = not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0
    if seeds_csv:
        _repair(seeds_csv)
    new_seeds = seeds_csv and (not os.path.isfile(seeds_csv) or os.path.getsize(seeds_csv) == 0)
    t0 = time.time()
    with open(out_csv, "a", newline="", encoding="utf-8") as f, \
            open(seeds_csv or os.devnull, "a", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        sw = csv.writer(fs)
        if new:
            w.writeheader()
        if new_seeds:
            sw.writerow(["file", "placement", "points", "notes"])   # points: [[row, col, type], ...]
        with Pool(workers) as pool:
            for n, (rows, seed_rows) in enumerate(pool.imap_unordered(process, tasks), 1):
                w.writerows(rows)
                if seeds_csv:
                    sw.writerows(seed_rows)
                f.flush()
                fs.flush()
                os.fsync(f.fileno())
                if n % 10 == 0 or n == len(tasks):
                    el = time.time() - t0
                    print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                          f"~{el/n*(len(tasks)-n)/60:.1f} min left", flush=True)


def environment():
    import numpy, scipy, skimage
    return {"cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--draw", action="store_true", help="write the 25% subsample (sample_25.csv)")
    ap.add_argument("--force", action="store_true", help="redraw an existing subsample")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=None, help="first N slices (smoke test)")
    ap.add_argument("--repeat", type=int, default=None, help="re-run the automatic arms as repeat k")
    args = ap.parse_args()

    if args.draw:
        write_sample(args.out, args.force)
        sys.exit(0)
    sample = read_sample(args.out)
    if args.limit:
        sample = sample[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"config": Config().to_dict(), "environment": environment()}, f, indent=2, default=str)
    if args.repeat:
        run_all(sample, os.path.join(args.out, f"raw_results_repeat{args.repeat}.csv"), AUTO_CONFIGS,
                args.workers)
    else:
        run_all(sample, os.path.join(args.out, "raw_results.csv"), args.configs, args.workers,
                os.path.join(args.out, "manual_seeds.csv"))
