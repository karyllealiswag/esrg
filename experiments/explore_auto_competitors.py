"""
explore_auto_competitors.py — EXPLORATORY side-test: automatic competing seeds (DEV only).

Purpose : See whether giving the methods automatically chosen competing seeds (healthy tissue,
          background), found from the image alone, changes performance. It is separate from
          ESRG: nothing in esrg/ is modified, the v2 parameters are not touched, and the result is
          descriptive (experiments/IMPROVEMENT_PROTOCOL_ESRG.md, §7).
Function : auto_competitors() places a background seed (the air pixel farthest from the brain) and a
          healthy seed (the typical-intensity deep brain pixel in the hemisphere opposite the
          automatic tumor seed, >= 25 mm from it) without the ground truth; process() runs
          ESRG v2 alone, ESRG v2 in compete mode and SRG with those seeds, plus planted competitors
          in compete mode as an upper reference; the ground truth is read only to report where
          the competitors landed and to score.
Notes   : The automatic tumor seed is seeds.select_seed with the BraTS-tuned seeding parameters
          (configs/esrg_brats_frozen.json). Arms already computed elsewhere (A_ESRG v1 and SRG with
          planted competitors, outputs/diagnostic_esrg_dev/) are reused by the analysis.
          CLI: python -P experiments/explore_auto_competitors.py
"""
import csv
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage import filters

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import preprocessing as pre
from esrg import seeds as seedmod
from esrg.io_utils import load_image, load_mask
from experiments.evaluate import seed_protocol, MM
from experiments.sampling import read_split

OUT = "outputs/explore_auto_competitors"
V2 = "configs/esrg_brats_frozen_v2.json"
HEALTHY_CLEAR = 25 * MM
FIELDS = ["file", "tumor", "arm", "status", "dsc", "recall", "precision", "seed_hit", "area_ratio",
          "retries", "healthy_in_gt", "bg_in_gt", "healthy_rule", "seconds"]


def auto_competitors(raw, cfg):
    """
    ([(row, col, 2), (row, col, 3)], info) from the image alone, or ([], info) when the automatic
    tumor seed finds nothing. Deterministic: ties go to the first pixel in row-major order.
    """
    img = pre.normalize(raw)
    mask, depth, _ = pre.head_mask(img, cfg)
    core, _ = seedmod.select_seed(np.where(mask, img, 0.0), mask, depth, cfg)
    if not core.any():
        return [], {"healthy_rule": "no tumor seed"}
    cr, cc = np.argwhere(core).mean(axis=0)
    # background: the air pixel farthest from the brain
    bg = np.unravel_index(int(np.argmax(ndi.distance_transform_edt(~mask))), mask.shape)
    # healthy tissue: typical-intensity deep brain pixel, opposite hemisphere, far from the seed
    cols = np.flatnonzero(mask.any(axis=0))
    mid = (cols[0] + cols[-1]) / 2
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    side = (xx < mid) if cc >= mid else (xx > mid)
    far = ndi.distance_transform_edt(~core) >= HEALTHY_CLEAR
    deep = mask & (depth > 0.4 * depth[mask].max())
    rule, pool = "contralateral", deep & side & far
    if not pool.any():
        rule, pool = "far (no contralateral pixel)", deep & far
    if not pool.any():
        rule, pool = "deep (anywhere)", deep
    smooth = filters.gaussian(np.where(mask, img, 0.0), sigma=3, preserve_range=True)
    target = float(np.median(img[deep]))
    cost = np.where(pool, np.abs(smooth - target), np.inf)
    h = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return [(int(h[0]), int(h[1]), 2), (int(bg[0]), int(bg[1]), 3)], {"healthy_rule": rule}


def process(task):
    ip, mp, info = task
    base = Config.load(V2)
    raw, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, raw.shape)
    pts, ainfo = auto_competitors(raw, base)
    planted = seed_protocol(raw, gt, base, info["file"])[0][1]
    arms = (("A_ESRG_v2", base.replace(seed_mode="auto"), None),
            ("A_ESRG_v2_autoC", base.replace(seed_mode="auto", esrg_compete=True), pts),
            ("A_SRG_autoC", base.replace(seed_mode="auto", method="srg"), pts),
            ("P_ESRG_v2_plantedC", base.replace(seed_mode="manual", esrg_compete=True), planted))
    rows = []
    for arm, cfg, points in arms:
        r = {"file": info["file"], "tumor": info["tumor"], "arm": arm, "healthy_rule": ainfo["healthy_rule"]}
        if pts and "autoC" in arm:
            r["healthy_in_gt"] = bool(gt[pts[0][0], pts[0][1]])
            r["bg_in_gt"] = bool(gt[pts[1][0], pts[1][1]])
        t0 = time.perf_counter()
        try:
            res = run(ip, cfg, mask_path=mp, manual_points=points)
            sc = res.scores
            r.update(status=res.status, dsc=sc.get("dsc") or 0.0, recall=sc.get("recall"),
                     precision=sc.get("precision"), seed_hit=sc.get("seed_hit"), area_ratio=sc.get("area_ratio"),
                     retries=res.stage("growth").info.get("flood guard retries", 0))
        except Exception as e:
            r.update(status="ERROR " + repr(e)[:80], dsc=0.0)
        r["seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(r)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rows.csv")
    done = set()
    if os.path.isfile(path):
        with open(path, newline="") as f:
            done = {r["file"] for r in csv.DictReader(f)}
    sample = [t for t in read_split("brats_eval", "dev") if t[2]["file"] not in done]
    print(f"{len(sample)} DEV slices to run", flush=True)
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    t0 = time.time()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        with Pool(11) as pool:
            for n, rows in enumerate(pool.imap_unordered(process, sample), 1):
                w.writerows(rows)
                f.flush()
                if n % 25 == 0 or n == len(sample):
                    el = time.time() - t0
                    print(f"  {n}/{len(sample)}  {el/60:.1f} min elapsed, ~{el/n*(len(sample)-n)/60:.1f} min left", flush=True)
