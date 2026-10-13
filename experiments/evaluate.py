"""
evaluate.py — Batch evaluation on BraTS 2023 GLI + MEN (experiments/PROTOCOL_BRATS.md).

Purpose : Run every arm of Experiments E1–E8 on one set of brats_eval/ (DEVELOPMENT
          for smoke tests and tuning checks, TEST exactly once with frozen
          parameters) and record one raw row per (slice, configuration).
Function : seed_protocol() places the planted seeds of a slice (tumor, healthy tissue,
          background; placements P1, P2, P3) deterministically and logs them;
          process() runs all configurations of CONFIGS on one slice; run_all()
          spreads slices over worker processes and appends rows to the raw CSV,
          skipping rows already present so an interrupted run resumes. --repeat
          re-runs the automatic arms into a separate file for the determinism check.
Notes   : Seeds (Section 6.2 of the protocol), at 512 px = 240 mm (1 mm = 2.133 px):
            tumor      core pixels >= 3 px from the core edge (the r = 3 disk fits);
                       P1 = deepest core pixel, P2/P3 = random pixels 4-8 mm from P1,
                       P3 also >= 4 mm from P2;
            healthy    brain pixels with depth > 0.4 max, >= 25 mm from the core, in
                       the hemisphere opposite the core centroid (edema masks are not
                       available; edema is mostly ipsilateral);
            background zero pixels >= 8 px outside the brain.
          Every arm of a comparison receives the same seeds. The ground truth places
          the planted seeds only (a careful operator); the automatic tumor seed
          never sees it. Parameters: configs/esrg_brats_frozen.json when present,
          otherwise the defaults with pre_stripped=True (smoke tests before tuning).
          CLI: python -P experiments/evaluate.py --set dev --limit 5      (smoke test)
               python -P experiments/evaluate.py --set test --sample-per-type 300 --approved --analyze
                                                                         (the one TEST run)
               ... --repeat 1 / --repeat 2 with the same --sample-per-type (determinism)
          TEST sample (PROTOCOL_BRATS §10): 300 patients per tumor type drawn from the TEST list by
          simple random sampling (files sorted, random.Random(2026).shuffle, first 300); the drawn
          list is written to <out>/test/test_sample.csv before any slice is run and reused after.
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
from experiments.sampling import BRATS_OUT, read_split

OUT = "outputs/evaluation_brats"
FROZEN = "configs/esrg_brats_frozen_v3.json"   # v2 + running global mean (IMPROVEMENT_PROTOCOL_ESRG_V3 §11)
SUCCESS_DSC = 0.70             # success threshold (Zijdenbos et al., 1994)
MM = 512 / 240                 # px per mm: BraTS 240 px at 1 mm, exported at 512 px
NEAR = (4 * MM, 8 * MM)        # P2/P3 lie 4-8 mm from P1
SEP = 4 * MM                   # P3 lies >= 4 mm from P2
HEALTHY_CLEAR = 25 * MM        # healthy seed >= 25 mm from the tumor core
BG_CLEAR = 8                   # background seed >= 8 px outside the brain

# id: (description, config overrides, seeding, placement, seed types used)
#   seeding "auto": automatic tumor seed + planted competitors (types 2-3);
#   seeding "manual": every planted type of the placement.
# SRG mirrors read ESRG's settings (v3 file has no separate SRG tuning): LLDM r, k_L, k_G, flood guard,
# and the sigma cap. The cap was calibrated in log units, so the ASC-only arms (no LLDM, intensity
# domain) omit it (IMPROVEMENT_PROTOCOL_ESRG_V3 §11); SRG is 8-connected as published.
NOCAP = dict(sigma_cap=None)
CONFIGS = {
    "A_ESRG":       ("ESRG, automatic seeding",                       dict(),                                        "auto", 1, (2, 3)),
    "A_SRG":        ("SRG, automatic tumor seed + planted competitors", dict(method="srg"),                        "auto", 1, (2, 3)),
    "P_SRG":        ("SRG (1994), planted P1",                        dict(method="srg"),                            "manual", 1, (1, 2, 3)),
    "P_SRG_2":      ("SRG (1994), planted P2",                        dict(method="srg"),                            "manual", 2, (1, 2, 3)),
    "P_SRG_3":      ("SRG (1994), planted P3",                        dict(method="srg"),                            "manual", 3, (1, 2, 3)),
    "P_ESRG":       ("ESRG, planted P1 tumor seed",                   dict(),                                        "manual", 1, (1, 2, 3)),
    "P_SRG_L":      ("SRG + LLDM, planted P1",                        dict(method="srg", srg_use_log_local=True),    "manual", 1, (1, 2, 3)),
    "P_ESRG_noL":   ("ESRG without LLDM (ASC on), planted P1",        dict(use_log=False, use_log_local=False, **NOCAP), "manual", 1, (1, 2, 3)),
    "P_SRG_solo":   ("SRG (1994), P1 tumor seed only",                dict(method="srg"),                            "manual", 1, (1,)),
    "P_SRG_solo_S": ("SRG + ASC, P1 tumor seed only",                 dict(method="srg", srg_use_stopping=True, **NOCAP), "manual", 1, (1,)),
    "P_ESRG_C":     ("ESRG compete mode (LLDM + ASC), planted P1",    dict(esrg_compete=True),                       "manual", 1, (1, 2, 3)),
    "P_ESRG_C_noS": ("ESRG compete mode without ASC, planted P1",     dict(esrg_compete=True, use_stopping=False),   "manual", 1, (1, 2, 3)),
}
AUTO_CONFIGS = [c for c, v in CONFIGS.items() if v[2] == "auto"]

STAGE_KEYS = ["input", "mask", "log", "candidates", "growth", "final"]
FIELDS = ["file", "tumor", "patient", "set", "config", "method", "seeding", "placement", "status",
          "tp", "fp", "fn", "pred_area", "gt_area",
          "dsc", "precision", "recall", "hd95", "hd95_mm", "hd95_wc", "hd95_wc_mm", "assd",
          "area_ratio", "leaked", "success", "dsc_enhancing",
          "seed_area", "seed_in_gt_frac", "seed_hit", "seed_distance", "mask_sha1",
          "tumor_seed_mm_from_p1", "sigma_floor", "sigma_A_initial", "stop_reason", "n_passes",
          "bucket", "bucket_reason", "seconds"] + [f"t_{k}" for k in STAGE_KEYS] + ["error"]
SCORE_KEYS = ("tp", "fp", "fn", "pred_area", "gt_area", "dsc", "precision", "recall",
              "hd95", "hd95_wc", "assd", "area_ratio", "leaked", "seed_hit", "seed_distance",
              "mask_sha1", "seconds")


def base_config():
    """Frozen BraTS parameters when tuning has been approved, else defaults (pre_stripped)."""
    if os.path.isfile(FROZEN):
        return Config.load(FROZEN).replace(pre_stripped=True)
    return Config(pre_stripped=True)


# ─── Seed protocol ───────────────────────────────────────────────────────────
def _rng(file, *parts):
    return random.Random(zlib.crc32("|".join([file, *map(str, parts)]).encode()))


def _near(pool, anchor, rng, avoid=None):
    """
    A pixel of pool ((N, 2) array) 4-8 mm from anchor (and >= 4 mm from avoid); if
    the ring is empty (small region), the pool pixel whose distance to anchor is
    closest to 6 mm. Returns (row, col).
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
        i = cand[np.argmin(np.abs(d[cand] - 6 * MM))]
    return int(pool[i][0]), int(pool[i][1])


def seed_protocol(raw, gt, cfg, file):
    """
    {placement: [(row, col, type), ...]} for P1-P3 plus a log of where each seed came
    from. Deterministic per file (crc32 of file, placement and type).
    """
    img = pre.normalize(raw)
    brain, depth, _ = pre.brain_mask(img)
    log = {}

    # Tumor (type 1)
    dt = ndi.distance_transform_edt(gt)
    tum = np.argwhere(dt >= cfg.manual_seed_radius)
    if not len(tum):
        tum = np.argwhere(gt)
    p1 = np.unravel_index(int(np.argmax(dt)), dt.shape)
    t2 = _near(tum, p1, _rng(file, "P2", 1))
    t3 = _near(tum, p1, _rng(file, "P3", 1), avoid=t2)

    # Healthy tissue (type 2): contralateral, deep, far from the core
    cols = np.flatnonzero(brain.any(axis=0))
    mid = (cols[0] + cols[-1]) / 2 if cols.size else gt.shape[1] / 2
    tumor_col = float(np.argwhere(gt)[:, 1].mean())
    side = np.zeros_like(brain)
    if tumor_col >= mid:
        side[:, :int(np.floor(mid))] = True
    else:
        side[:, int(np.ceil(mid)) + 1:] = True
    away = ndi.distance_transform_edt(~gt)
    deep = brain & (depth > 0.4 * depth.max()) & ~gt
    healthy = np.argwhere(deep & (away >= HEALTHY_CLEAR) & side)
    log["healthy_rule"] = "contralateral"
    if not len(healthy):
        healthy = np.argwhere(deep & (away >= HEALTHY_CLEAR))
        log["healthy_rule"] = "far (no contralateral pixel)"
    if not len(healthy):
        far = np.where(deep, away, -1)
        healthy = np.argwhere(far == far.max())
        log["healthy_rule"] = "farthest deep brain pixel"
    rng = _rng(file, "P1", 2)
    h1 = tuple(int(v) for v in healthy[rng.randrange(len(healthy))])
    h2 = _near(healthy, h1, _rng(file, "P2", 2))
    h3 = _near(healthy, h1, _rng(file, "P3", 2), avoid=h2)

    # Background (type 3)
    bg = np.argwhere((img == 0) & (ndi.distance_transform_edt(~brain) >= BG_CLEAR))
    rng = _rng(file, "P1", 3)
    b1 = tuple(int(v) for v in bg[rng.randrange(len(bg))])
    b2 = _near(bg, b1, _rng(file, "P2", 3))
    b3 = _near(bg, b1, _rng(file, "P3", 3), avoid=b2)

    p1 = (int(p1[0]), int(p1[1]))
    seeds = {1: [(*p1, 1), (*h1, 2), (*b1, 3)],
             2: [(*t2, 1), (*h2, 2), (*b2, 3)],
             3: [(*t3, 1), (*h3, 2), (*b3, 3)]}
    log["tumor_mm_from_p1"] = {k: round(float(np.hypot(s[0][0] - p1[0], s[0][1] - p1[1])) / MM, 2)
                               for k, s in seeds.items()}
    return seeds, log


# ─── One slice, all configurations ───────────────────────────────────────────
def _row_from_result(row, res, gt, enh, cfg):
    sc = res.scores
    row.update({k: sc.get(k) for k in SCORE_KEYS})
    for k in ("hd95", "hd95_wc"):
        row[f"{k}_mm"] = None if sc.get(k) is None or np.isnan(sc[k]) else round(sc[k] / MM, 4)
    row["status"] = res.status
    row["success"] = bool(sc.get("dsc") is not None and sc["dsc"] >= SUCCESS_DSC)
    if row["tumor"] == "glioma" and enh is not None:          # secondary target (protocol §1)
        p = res.mask.astype(bool)
        den = p.sum() + enh.sum()
        row["dsc_enhancing"] = float(2 * (p & enh).sum() / den) if den else None
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
    ip, mp, info, cfg_ids, base_kw = task
    base = Config(**base_kw)
    raw, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, raw.shape)
    ep = mp.replace("/masks/", "/masks_enhancing/")
    enh = load_mask(ep, raw.shape) if os.path.isfile(ep) else None
    seeds, slog = seed_protocol(raw, gt, base, info["file"])
    rows = []
    for cid in cfg_ids:
        desc, over, seeding, placement, types = CONFIGS[cid]
        cfg = base.replace(seed_mode=seeding, **over)
        row = {"file": info["file"], "tumor": info["tumor"], "patient": info["patient"], "set": info["set"],
               "config": cid, "method": cfg.method, "seeding": seeding, "placement": placement,
               "tumor_seed_mm_from_p1": slog["tumor_mm_from_p1"][placement] if 1 in types else None}
        try:
            points = [p for p in seeds[placement] if p[2] in types]
            res = run(ip, cfg, mask_path=mp, manual_points=points)
            _row_from_result(row, res, gt, enh, cfg)
            if cid == "A_ESRG":
                row["bucket"], row["bucket_reason"] = attribute(res)
        except Exception as e:                       # recorded, never silently dropped
            row["status"], row["error"] = "ERROR", repr(e)
        rows.append(row)
    seed_rows = [[info["file"], k, json.dumps(v), slog["healthy_rule"]] for k, v in seeds.items()]
    return rows, seed_rows


def _done_keys(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["file"], r["config"]) for r in csv.DictReader(f)}


def run_all(sample, out_csv, cfg_ids, workers, base_kw):
    done = _done_keys(out_csv)
    tasks = []
    for ip, mp, info in sample:
        todo = [c for c in cfg_ids if (info["file"], c) not in done]
        if todo:
            tasks.append((ip, mp, info, todo, base_kw))
    print(f"{len(sample)} slices, {len(cfg_ids)} configurations; {len(tasks)} slices still to run", flush=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    new = not os.path.isfile(out_csv)
    seeds_csv = os.path.join(os.path.dirname(out_csv), "manual_seeds.csv")
    new_seeds = not os.path.isfile(seeds_csv)
    t0 = time.time()
    with open(out_csv, "a", newline="", encoding="utf-8") as f, \
            open(seeds_csv, "a", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        sw = csv.writer(fs)
        if new:
            w.writeheader()
        if new_seeds:
            sw.writerow(["file", "placement", "points", "healthy_rule"])   # points: [[row, col, type], ...]
        with Pool(workers) as pool:
            for n, (rows, seed_rows) in enumerate(pool.imap_unordered(process, tasks), 1):
                w.writerows(rows)
                sw.writerows(seed_rows)
                f.flush()
                fs.flush()
                if n % 25 == 0 or n == len(tasks):
                    el = time.time() - t0
                    print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                          f"~{el/n*(len(tasks)-n)/60:.1f} min left", flush=True)


def draw_sample(sample, n, manifest, seed=2026):
    """Simple random sample of n patients per tumor type (files sorted, then shuffled with a fixed
    seed). The list is written once; later calls (resume, repeats) must reproduce it exactly."""
    out = []
    for k in ("glioma", "meningioma"):
        pool = sorted((t for t in sample if t[2]["tumor"] == k), key=lambda t: t[2]["file"])
        random.Random(seed).shuffle(pool)
        out += pool[:n]
    files = [t[2]["file"] for t in out]
    if os.path.isfile(manifest):
        old = [r["file"] for r in csv.DictReader(open(manifest))]
        if old != files:
            raise SystemExit(f"{manifest} differs from the draw: refusing to run on a different sample.")
    else:
        with open(manifest, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["order", "tumor", "patient", "file"])
            for i, t in enumerate(out):
                w.writerow([i, t[2]["tumor"], t[2]["patient"], t[2]["file"]])
    return out


def environment():
    """Hardware/software of the run (reported with the results)."""
    import numpy, scipy, skimage
    return {"cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("dev", "test"), required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--limit", type=int, default=None, help="first N slices of the set (smoke test)")
    ap.add_argument("--per-type", type=int, default=None, help="first N slices of each tumor type (smoke test)")
    ap.add_argument("--repeat", type=int, default=None, help="re-run the automatic arms as repeat k")
    ap.add_argument("--part", choices=("D1", "D2", "D3"), default=None,
                    help="dev only: restrict to one part of the v3 DEV split (outputs/improve_v3/split.csv)")
    ap.add_argument("--sample-per-type", type=int, default=None,
                    help="TEST: random sample of N patients per tumor type (seed 2026), recorded in test_sample.csv")
    ap.add_argument("--analyze", action="store_true", help="run analyze_brats.py on the folder afterwards")
    ap.add_argument("--approved", action="store_true",
                    help="TEST only: the user has approved the frozen parameters and the single TEST run")
    args = ap.parse_args()

    if args.set == "test":
        if not os.path.isfile(FROZEN):
            raise SystemExit(f"{FROZEN} not found: TEST runs only with the frozen, approved parameters.")
        if not args.approved:
            raise SystemExit("TEST is a single run and needs the user's explicit approval: pass --approved.")
    sample = read_split(BRATS_OUT, args.set)
    if args.part:
        if args.set != "dev":
            raise SystemExit("--part applies to the DEV set only")
        part = dict(csv.reader(open("outputs/improve_v3/split.csv")))
        sample = [t for t in sample if part[t[2]["file"]] == args.part]
    if args.per_type:
        sample = [t for k in ("glioma", "meningioma")
                  for t in [x for x in sample if x[2]["tumor"] == k][:args.per_type]]
    if args.limit:
        sample = sample[:args.limit]
    base = base_config()
    out_dir = os.path.join(args.out, args.set)
    os.makedirs(out_dir, exist_ok=True)
    if args.sample_per_type:
        sample = draw_sample(sample, args.sample_per_type, os.path.join(out_dir, f"{args.set}_sample.csv"))
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"config": base.to_dict(), "frozen_file": FROZEN if os.path.isfile(FROZEN) else None,
                   "environment": environment()}, f, indent=2, default=str)
    if args.repeat:
        run_all(sample, os.path.join(out_dir, f"raw_results_repeat{args.repeat}.csv"), AUTO_CONFIGS,
                args.workers, base.to_dict())
    else:
        run_all(sample, os.path.join(out_dir, "raw_results.csv"), args.configs, args.workers, base.to_dict())
    if args.analyze:
        from experiments.analyze_brats import analyze
        analyze(out_dir)
