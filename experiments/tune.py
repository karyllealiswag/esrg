"""
tune.py — Pre-specified parameter search on the BraTS DEVELOPMENT set (PROTOCOL_BRATS §5).

Purpose : Choose the BraTS parameters of ESRG and of the SRG modules on brats_eval/dev
          only, log every configuration tried, and freeze the winners for one TEST run.
Function : t2() — automatic seeding (shared by A_ESRG and A_SRG): 972 grid points
          scored by seed hit rate. Per slice, everything that does not depend on a
          grid point is computed once: the brain mask and depth, then for each of
          the 18 (otsu_classes, interior_frac, min_clearance) combinations the
          Otsu candidates and every candidate's area, radius, clearance, contrast
          and shape. The remaining parameters only filter and re-score those
          cached candidates, exactly as seeds.select_seed and
          seed_ranking.score_candidates do (verify_t2 checks this against the real
          functions). t1() — growth: ESRG (145), SRG + LLDM (5), SRG + ASC (29) and
          the 1994 SRG (1) from the planted P1 seeds, scored by DSC.
          summarize() averages per patient, per tumor type, then over the two types,
          applies the tie-breaks, writes the sensitivity tables and freezes
          configs/esrg_brats_frozen.json.
Notes   : Crash-safe and resumable: every finished task appends its rows to
          outputs/tuning_brats/*.csv and flushes; on restart a truncated last line is
          cut off and finished (configuration, slice) pairs are skipped, so rerunning
          the same command continues where it stopped and never loses rows.
          CLI: python -P experiments/tune.py --stage t2      (seeding; then t1)
               python -P experiments/tune.py --stage t1
               python -P experiments/tune.py --verify-t2      (cache = select_seed check)
               python -P experiments/tune.py --summarize      (winners, sensitivity, freeze)
"""
import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import preprocessing as pre
from esrg import seed_ranking, seeds
from esrg.io_utils import load_image, load_mask
from experiments.evaluate import seed_protocol
from experiments.sampling import BRATS_OUT, SEED, read_split

OUT = "outputs/tuning_brats"
FROZEN = "configs/esrg_brats_frozen.json"
BASE = Config(pre_stripped=True)          # default values = the tie-break reference
CAP_HOURS = 12.0                          # per-stage wall-time cap (protocol §5.3)

# ─── Grids (protocol §5.3) ───────────────────────────────────────────────────
K_PAIRS = [(kl, kg) for kl in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
           for kg in (2.0, 2.5, 3.0, 4.0, 5.0) if kg > kl]                    # 29 pairs
RADII = (1, 2, 3, 4, 5)
# Deviation (protocol §10, Checkpoint 3): the SRG + ASC and SRG + LLDM winners sat at the
# top edge of the pre-specified grid and were still rising, so their grids are extended
# upward on DEV only. ESRG's grid is unchanged.
SRGS_EXT = ([(3.0, kg) for kg in (6.0, 7.0, 8.0)]
            + [(kl, kg) for kl in (3.5, 4.0, 5.0, 6.0) for kg in (4.0, 5.0, 6.0, 7.0, 8.0) if kg > kl])  # 17
SRGL_EXT = (6, 7, 8, 10)
T2_GRID = {"otsu_classes": (3, 4), "seed_core_frac": (0.5, 0.7, 0.85), "min_cand_area": (20, 40, 80),
           "interior_frac": (0.0, 0.06, 0.12), "min_clearance": (0.0, 4.0, 8.0),
           "rank_w_contrast": (0.25, 0.5, 1.0), "rank_w_shape": (0.0, 0.2)}   # 972 points


def t1_configs():
    """{config id: (arm, overrides)} for growth tuning."""
    out = {}
    for (kl, kg), r in itertools.product(K_PAIRS, RADII):
        out[f"ESRG_kl{kl}_kg{kg}_r{r}"] = ("ESRG", dict(k_local=kl, k_global=kg, local_radius=r))
    for r in RADII + SRGL_EXT:
        out[f"SRGL_r{r}"] = ("SRG+L", dict(method="srg", srg_use_log_local=True, srg_local_radius=r))
    for kl, kg in K_PAIRS + SRGS_EXT:
        out[f"SRGS_kl{kl}_kg{kg}"] = ("SRG+S", dict(method="srg", srg_use_stopping=True,
                                                     srg_k_local=kl, srg_k_global=kg))
    out["SRG"] = ("SRG", dict(method="srg"))
    return out


def t2_points():
    keys = list(T2_GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*T2_GRID.values())]


def point_id(p):
    return "_".join(f"{k}={p[k]:g}" for k in T2_GRID)


# ─── Crash-safe append log ───────────────────────────────────────────────────
def _repair(path):
    """Cut a truncated last line (a crash mid-write) so the file parses cleanly."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def _done(path, key_fields):
    _repair(path)
    if not os.path.isfile(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {tuple(r[k] for k in key_fields) for r in csv.DictReader(f)}


def _drive(tasks, worker, path, fields, workers, label):
    """Run tasks in a pool; append and flush each task's rows the moment it finishes."""
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    t0 = time.time()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
            f.flush()
        with Pool(workers) as pool:
            for n, rows in enumerate(pool.imap_unordered(worker, tasks), 1):
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
                if n % 10 == 0 or n == len(tasks):
                    el = time.time() - t0
                    print(f"  {label} {n}/{len(tasks)} tasks  {el/60:.1f} min elapsed, "
                          f"~{el/n*(len(tasks)-n)/60:.1f} min left", flush=True)
                if (time.time() - t0) / 3600 > CAP_HOURS:
                    print(f"  {label}: {CAP_HOURS} h cap reached; stopping (rerun to continue).", flush=True)
                    pool.terminate()
                    return


# ─── T2: automatic seeding, cached ───────────────────────────────────────────
def _components(img, mask, depth, otsu_classes, interior_frac, min_clearance, cfg):
    """
    seeds.select_seed up to the candidate list, for one (K, eta, c) combination:
    returns (status, [(rec, comp, dt, contrast, compactness, solidity), ...]) for
    every component, before the area / clearance filters.
    """
    depth_cut = max(min_clearance, interior_frac * depth.max())
    interior = mask & (depth >= depth_cut)
    if interior.sum() < 50:
        interior = mask
    vals = img[interior]
    if vals.size < 50 or vals.max() - vals.min() < 1e-6:
        return "flat", []
    try:
        thresholds = filters.threshold_multiotsu(vals.reshape(-1, 1), classes=otsu_classes)
    except ValueError:
        thresholds = np.array([filters.threshold_otsu(vals)])
    cand = interior & (img >= thresholds[-1])
    cand = morphology.opening(cand, morphology.disk(1))
    lab, n = ndi.label(cand)
    comps = []
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < min(T2_GRID["min_cand_area"]):
            continue                            # below every grid value: never kept
        dt = ndi.distance_transform_edt(comp)
        idx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        radius = float(dt[idx])
        clearance = float(depth[idx]) - radius
        contrast = seed_ranking._contrast(comp, img, cfg.rank_contrast_ring)
        compactness, solidity = seed_ranking._shape(comp)
        comps.append(({"id": i, "area": area, "radius": radius, "clearance": clearance},
                      comp, dt, contrast, compactness, solidity))
    return "ok", comps


def _choose(comps, p, cfg):
    """The remaining parameters: filters, then score_candidates' exact arithmetic."""
    kept = [c for c in comps if c[0]["area"] >= p["min_cand_area"] and c[0]["clearance"] >= p["min_clearance"]]
    if not kept:
        return None
    radius = np.array([float(c[2].max()) for c in kept], float)
    contrast = np.array([c[3] for c in kept], float)

    def norm(vals):
        lo, hi = vals.min(), vals.max()
        return (vals - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(vals)

    nr, nc = norm(radius), norm(contrast)
    ranked = []
    for c, r_, c_ in zip(kept, nr, nc):
        shape_term = 0.5 * (c[4] + c[5])
        score = (p["rank_w_contrast"] * c_ + cfg.rank_w_radius * r_ + p["rank_w_shape"] * shape_term)
        ranked.append((score, c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    comp, dt = ranked[0][1][1], ranked[0][1][2]
    core = comp & (dt >= p["seed_core_frac"] * dt.max())
    if not core.any():
        core = comp & (dt >= dt.max())
    return core


def _slice_inputs(ip, mp, cfg):
    raw, _ = load_image(ip, cfg.max_side)
    img, _ = pre.normalize_details(raw)
    mask, depth, _ = pre.head_mask(img, cfg)
    return np.where(mask, img, 0.0), mask, depth, load_mask(mp, img.shape)


def t2_worker(task):
    ip, mp, info = task
    cfg = BASE
    img, mask, depth, gt = _slice_inputs(ip, mp, cfg)
    t0 = time.perf_counter()
    cache = {}
    for K in T2_GRID["otsu_classes"]:
        for eta in T2_GRID["interior_frac"]:
            for c in T2_GRID["min_clearance"]:
                cache[(K, eta, c)] = _components(img, mask, depth, K, eta, c, cfg)
    rows = []
    for p in t2_points():
        status, comps = cache[(p["otsu_classes"], p["interior_frac"], p["min_clearance"])]
        core = _choose(comps, p, cfg) if status == "ok" else None
        hit = bool(core is not None and core.any() and not (core & ~gt).any())
        rows.append({"point": point_id(p), "file": info["file"], "tumor": info["tumor"],
                     "patient": info["patient"], **p, "seeded": core is not None,
                     "hit": hit, "core_area": int(core.sum()) if core is not None else 0})
    sec = (time.perf_counter() - t0) / len(rows)
    for r in rows:
        r["seconds"] = round(sec, 6)
    return rows


T2_FIELDS = ["point", "file", "tumor", "patient", *T2_GRID, "seeded", "hit", "core_area", "seconds"]


def t2(workers):
    path = os.path.join(OUT, "t2_seeding.csv")
    done = _done(path, ("point", "file"))
    n_pts = len(t2_points())
    sample = read_split(BRATS_OUT, "dev")
    tasks = [t for t in sample
             if sum(1 for p in t2_points() if (point_id(p), t[2]["file"]) in done) < n_pts] if done else sample
    print(f"T2: {len(sample)} DEV slices x {n_pts} grid points; {len(tasks)} slices to run", flush=True)
    _drive(tasks, t2_worker, path, T2_FIELDS, workers, "T2")


def verify_t2(n=300, seed=SEED):
    """The cache must reproduce seeds.select_seed exactly: compare cores on random (slice, point) pairs."""
    rng = random.Random(seed)
    sample = read_split(BRATS_OUT, "dev")
    pts = t2_points()
    mismatches = 0
    for k in range(n):
        ip, mp, info = rng.choice(sample)
        p = rng.choice(pts)
        cfg = BASE.replace(**p)
        img, mask, depth, gt = _slice_inputs(ip, mp, cfg)
        real, _ = seeds.select_seed(img, mask, depth, cfg)
        status, comps = _components(img, mask, depth, p["otsu_classes"], p["interior_frac"], p["min_clearance"], cfg)
        mine = _choose(comps, p, cfg) if status == "ok" else None
        mine = mine if mine is not None else np.zeros_like(real)
        if not np.array_equal(real, mine):
            mismatches += 1
            print(f"  MISMATCH {info['file']} {point_id(p)}", flush=True)
    print(f"verify_t2: {n - mismatches}/{n} identical to seeds.select_seed", flush=True)
    return mismatches


# ─── T1: growth ──────────────────────────────────────────────────────────────
T1_CHUNK = 12                             # configurations per task (~20-30 s): fine-grained resume


def t1_worker(task):
    ip, mp, info, ids = task
    raw, _ = load_image(ip, BASE.max_side)
    gt = load_mask(mp, raw.shape)
    seeds_, _ = seed_protocol(raw, gt, BASE, info["file"])
    configs = t1_configs()
    rows = []
    for cid in ids:
        arm, over = configs[cid]
        cfg = BASE.replace(seed_mode="manual", **over)
        t0 = time.perf_counter()
        try:
            res = run(ip, cfg, mask_path=mp, manual_points=seeds_[1])
            sc, err = res.scores, None
        except Exception as e:
            sc, err = {}, repr(e)
        rows.append({"config": cid, "arm": arm, "file": info["file"], "tumor": info["tumor"],
                     "patient": info["patient"], "k_local": over.get("k_local", over.get("srg_k_local")),
                     "k_global": over.get("k_global", over.get("srg_k_global")),
                     "local_radius": over.get("local_radius", over.get("srg_local_radius")),
                     "dsc": sc.get("dsc", 0.0) if sc.get("dsc") is not None else 0.0,
                     "recall": sc.get("recall"), "precision": sc.get("precision"),
                     "seconds": round(time.perf_counter() - t0, 3), "error": err})
    return rows


T1_FIELDS = ["config", "arm", "file", "tumor", "patient", "k_local", "k_global", "local_radius",
             "dsc", "recall", "precision", "seconds", "error"]


def t1(workers):
    path = os.path.join(OUT, "t1_growth.csv")
    done = _done(path, ("config", "file"))
    ids = list(t1_configs())
    tasks = []
    for ip, mp, info in read_split(BRATS_OUT, "dev"):
        todo = [c for c in ids if (c, info["file"]) not in done]
        for i in range(0, len(todo), T1_CHUNK):
            tasks.append((ip, mp, info, todo[i:i + T1_CHUNK]))
    # Interleave slices so early progress covers the whole DEV set evenly.
    tasks.sort(key=lambda t: (ids.index(t[3][0]), t[2]["file"]))
    print(f"T1: {len(ids)} configurations x 518 DEV slices; {len(tasks)} tasks to run", flush=True)
    _drive(tasks, t1_worker, path, T1_FIELDS, workers, "T1")


# ─── Summary: winners, tie-breaks, sensitivity, freeze ───────────────────────
def _score_table(df, keys, value):
    """Per configuration: mean of `value` per tumor type (patient = slice), then over the types."""
    import pandas as pd
    per = df.groupby(keys + ["tumor"])[value].agg(["mean", "median", "count"]).unstack("tumor")
    out = pd.DataFrame(index=per.index)
    for t in ("glioma", "meningioma"):
        out[f"{t}_mean"] = per[("mean", t)]
        out[f"{t}_median"] = per[("median", t)]
        out[f"{t}_n"] = per[("count", t)]
    out["objective"] = (out["glioma_mean"] + out["meningioma_mean"]) / 2
    return out.reset_index()


def _pick(tab, params, base, grid, tol, last):
    """
    Best objective, then within tol of it (protocol §5.3): fewer parameters changed from
    the default values, smaller normalised distance from them, then `last` (column, ascending).
    """
    best = tab["objective"].max()
    near = tab[tab["objective"] >= best - tol].copy()
    near["n_changed"] = sum((near[p] != base[p]).astype(int) for p in params)
    near["distance"] = sum((near[p] - base[p]).abs() / (max(grid[p]) - min(grid[p])) for p in params)
    col, asc = last
    near = near.sort_values(["n_changed", "distance", col], ascending=[True, True, asc], kind="mergesort")
    return near.iloc[0], best, len(near)


def _sensitivity(tab, params, win):
    """Objective vs each parameter, the others held at the winner."""
    rows = []
    for p in params:
        others = [q for q in params if q != p]
        sub = tab
        for q in others:
            sub = sub[sub[q] == win[q]]
        for _, r in sub.sort_values(p).iterrows():
            rows.append({"parameter": p, "value": r[p], "objective": r["objective"],
                         "glioma_mean": r["glioma_mean"], "meningioma_mean": r["meningioma_mean"],
                         "is_winner": r[p] == win[p]})
    return rows


def summarize():
    import pandas as pd
    res = {}
    t2p = os.path.join(OUT, "t2_seeding.csv")
    t1p = os.path.join(OUT, "t1_growth.csv")
    _repair(t2p), _repair(t1p)
    frozen = {}

    # T2: seed hit rate
    d2 = pd.read_csv(t2p)
    params2 = list(T2_GRID)
    tab2 = _score_table(d2.assign(hit=d2.hit.astype(float), seeded=d2.seeded.astype(float)), params2, "hit")
    nc = d2.groupby(params2)["seeded"].apply(lambda v: 1 - v.astype(float).mean()).rename("no_candidate_rate")
    tab2 = tab2.merge(nc.reset_index(), on=params2)
    base2 = {p: getattr(BASE, p) for p in params2}
    win2, best2, n_tied2 = _pick(tab2, params2, base2, T2_GRID, 0.005, ("min_clearance", False))
    tab2.sort_values("objective", ascending=False).to_csv(os.path.join(OUT, "t2_configs.csv"), index=False)
    pd.DataFrame(_sensitivity(tab2, params2, win2)).to_csv(os.path.join(OUT, "t2_sensitivity.csv"), index=False)
    frozen.update({p: (int(win2[p]) if p in ("otsu_classes", "min_cand_area") else float(win2[p])) for p in params2})
    res["t2"] = {"n_points": len(tab2), "n_slices": int(d2.file.nunique()), "best_objective": float(best2),
                 "tied_within_tol": n_tied2, "winner": {p: frozen[p] for p in params2},
                 "winner_scores": {k: float(win2[k]) for k in ("objective", "glioma_mean", "meningioma_mean",
                                                               "no_candidate_rate")},
                 "default_values": base2,
                 "default_scores": tab2[(tab2[params2] == pd.Series(base2)).all(axis=1)][
                     ["objective", "glioma_mean", "meningioma_mean"]].to_dict("records")}

    # T1: DSC per arm
    d1 = pd.read_csv(t1p)
    res["t1_errors"] = int(d1.error.notna().sum())
    grids1 = {"ESRG": {"k_local": [k for k, _ in K_PAIRS], "k_global": [k for _, k in K_PAIRS],
                       "local_radius": list(RADII)},
              "SRG+L": {"local_radius": list(RADII + SRGL_EXT)},
              "SRG+S": {"k_local": [k for k, _ in K_PAIRS + SRGS_EXT], "k_global": [k for _, k in K_PAIRS + SRGS_EXT]}}
    res["t1"] = {}
    for arm, params, fields in (("ESRG", ["k_local", "k_global", "local_radius"], ("k_local", "k_global", "local_radius")),
                                ("SRG+L", ["local_radius"], ("srg_local_radius",)),
                                ("SRG+S", ["k_local", "k_global"], ("srg_k_local", "srg_k_global"))):
        sub = d1[d1.arm == arm]
        tab = _score_table(sub, params, "dsc")
        base1 = {"k_local": BASE.k_local, "k_global": BASE.k_global, "local_radius": BASE.local_radius}
        win, best, n_tied = _pick(tab, params, base1, grids1[arm], 0.005,
                                  ("k_local", True) if "k_local" in params else ("local_radius", True))
        tag = arm.replace("+", "")
        tab.sort_values("objective", ascending=False).to_csv(os.path.join(OUT, f"t1_{tag}_configs.csv"), index=False)
        pd.DataFrame(_sensitivity(tab, params, win)).to_csv(os.path.join(OUT, f"t1_{tag}_sensitivity.csv"), index=False)
        for p, f in zip(params, fields):
            frozen[f] = int(win[p]) if p == "local_radius" else float(win[p])
        res["t1"][arm] = {"n_configs": len(tab), "n_slices": int(sub.file.nunique()), "best_objective": float(best),
                          "tied_within_tol": n_tied, "winner": {p: float(win[p]) for p in params},
                          "winner_scores": {k: float(win[k]) for k in ("objective", "glioma_mean", "meningioma_mean",
                                                                       "glioma_median", "meningioma_median")}}
    srg = d1[d1.arm == "SRG"]
    res["t1"]["SRG"] = {"objective": float((srg[srg.tumor == "glioma"].dsc.mean() + srg[srg.tumor == "meningioma"].dsc.mean()) / 2),
                        "glioma_mean": float(srg[srg.tumor == "glioma"].dsc.mean()),
                        "meningioma_mean": float(srg[srg.tumor == "meningioma"].dsc.mean()),
                        "n_slices": int(srg.file.nunique())}

    cfg = Config(pre_stripped=True, **frozen).validate()
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    cfg.save(FROZEN)
    res["frozen_file"] = FROZEN
    res["frozen_changes_from_default"] = {k: v for k, v in cfg.to_dict().items() if v != BASE.to_dict().get(k)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("t1", "t2", "all"))
    ap.add_argument("--verify-t2", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="winners, sensitivity tables, frozen config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.verify_t2:
        sys.exit(1 if verify_t2() else 0)
    if args.summarize:
        summarize()
        sys.exit(0)
    if args.stage in ("t2", "all"):
        t2(args.workers)
    if args.stage in ("t1", "all"):
        t1(args.workers)
