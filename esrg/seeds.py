"""
seeds.py — Objective 1: automatic, deterministic seed selection.

Purpose : Locate the tumor and place a seed core without human input, or fail
          loudly ("no tumor candidate") rather than guess.
Function : Multi-Otsu keeps the brightest interior class; components are filtered by
          area and by clearance from the head outline (scalp sits on the rim); the
          survivors are ranked (seed_ranking) and the winner's distance-transform
          core becomes the seed. Anatomical exclusion (orbits/skull base) can be
          applied first as an ablation.
Notes   : The histogram uses interior pixels only, since scalp fat outshines an
          enhancing tumor on T1 and would otherwise capture the top Otsu class.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology
from . import seed_ranking
from . import anatomy_exclusion


def select_seed(img, mask, depth, cfg):
    """
    img   : normalized slice
    mask  : head mask from preprocessing.head_mask
    depth : distance from the head outline
    Returns (core, info). core is an empty bool array when no candidate survives.
    """
    info = {"thresholds": None, "n_raw": 0, "n_kept": 0, "components": [],
            "chosen": None, "core_area": 0, "status": "OK", "warnings": []}
    empty = np.zeros_like(mask, bool)

    # Restrict the histogram to interior tissue. Scalp fat outshines an enhancing
    # tumor on T1, so including it pushes the tumor out of the top Otsu class.
    interior = mask & (depth >= max(cfg.min_clearance, cfg.interior_frac * depth.max()))
    if interior.sum() < 50:
        interior = mask
    info["interior_frac_of_head"] = round(float(interior.sum() / max(mask.sum(), 1)), 2)
    vals = img[interior]
    if vals.size < 50 or vals.max() - vals.min() < 1e-6:
        info["status"] = "NO TUMOR CANDIDATE"
        info["warnings"].append("Head region empty or flat.")
        return empty, info

    # ── Multi-level Otsu; the enhancing tumor sits in the top class on T1-CE ──
    try:
        thresholds = filters.threshold_multiotsu(vals.reshape(-1, 1), classes=cfg.otsu_classes)
    except ValueError:
        thresholds = np.array([filters.threshold_otsu(vals)])
        info["warnings"].append("Multi-Otsu failed; fell back to single Otsu.")
    info["thresholds"] = [round(float(t), 1) for t in thresholds]

    cand = interior & (img >= thresholds[-1])
    cand = morphology.opening(cand, morphology.disk(1))          # drop 1-px speckle

    # Phase 2a: remove orbits / skull base BEFORE candidate labeling, since these
    # are high-contrast and cannot be out-ranked (Phase 1). Excludes by anatomical
    # signature (bright + round + upper region), not fixed coordinates.
    forbidden = anatomy_exclusion.forbidden_region(img, mask, depth, cfg)
    if forbidden.any():
        cand = cand & ~forbidden
        info["excluded_px"] = int(forbidden.sum())

    lab, n = ndi.label(cand)
    info["n_raw"] = int(n)
    if n == 0:
        info["status"] = "NO TUMOR CANDIDATE"
        info["warnings"].append("Top intensity class is empty after opening.")
        return empty, info

    # ── Interior filtering ───────────────────────────────────────────────────
    # Scalp fat is the brightest tissue on T1, so the top class is dominated by
    # a fat ring hugging the head outline. Geometry separates it from a tumor:
    # a component touching the outline has its deepest point at roughly its own
    # inscribed radius, while an interior component lies further in than that.
    # clearance = depth(center) - radius  is near zero for scalp, large for tumor.
    kept = []
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < cfg.min_cand_area:
            continue                                             # too small to be a tumor
        dt = ndi.distance_transform_edt(comp)
        idx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        radius = float(dt[idx])
        clearance = float(depth[idx]) - radius
        rec = {"id": i, "area": area, "radius": round(radius, 1),
               "clearance": round(clearance, 1),
               "mean": round(float(img[comp].mean()), 1)}
        if clearance < cfg.min_clearance:
            rec["rejected"] = "touches head outline"
            info["components"].append(rec)
            continue
        kept.append((rec, comp, dt))
        info["components"].append(rec)

    info["n_kept"] = len(kept)
    if not kept:
        # Hard fail, no fallback: an arbitrary seed would fabricate a detection.
        info["status"] = "NO TUMOR CANDIDATE"
        info["warnings"].append(
            f"All {n} candidates rejected (area < {cfg.min_cand_area} px or "
            f"clearance < {cfg.min_clearance} px from the head outline).")
        return empty, info

    # ── Rank candidates (Phase 1: contrast + vesselness + shape) ─────────────
    # Falls back to the original size-based pick if all ranking features are off.
    if cfg.rank_use_contrast or cfg.rank_use_vesselness or cfg.rank_use_shape:
        kept = seed_ranking.score_candidates(kept, img, mask, cfg)
        rec, comp, dt = kept[0]
    else:
        rec, comp, dt = max(kept, key=lambda k: (k[0]["radius"], k[0]["mean"]))
    info["chosen"] = rec
    info["ranked"] = [k[0] for k in kept[:6]]

    # ── Core = medial pixels, farthest from partial-volume boundary pixels ───
    core = comp & (dt >= cfg.seed_core_frac * dt.max())
    if not core.any():
        core = comp & (dt >= dt.max())                           # single deepest pixel
    info["core_area"] = int(core.sum())
    return core, info


def purify_core(core, cfg):
    """
    Erode a seed core to its own medial pixels (mirrors the dt >= seed_core_frac *
    dt.max() clip select_seed() already applies to the auto core), so growth's
    initial Welford stats aren't seeded from partial-volume boundary pixels.

    Used to bring a raw manual click disk up to the same purity guarantee the
    auto core gets for free from its shape-aware distance transform.
    """
    if not core.any():
        return core
    dt = ndi.distance_transform_edt(core)
    clipped = core & (dt >= cfg.seed_core_frac * dt.max())
    return clipped if clipped.any() else (core & (dt >= dt.max()))


def manual_seed(points, shape, cfg):
    """
    Build the seed map from user clicks (baseline comparison / manual mode).

    points are (r, c) or (r, c, type). Type 1 is the tumor; types 2..n are the
    competing regions the classical SRG tessellates against. Returns (labels, info)
    where labels == 1 is the tumor core.
    """
    labels = np.zeros(shape, np.int32)
    clicks = {}
    for p in points:
        r, c = p[0], p[1]
        t = int(p[2]) if len(p) > 2 else 1
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            clicks.setdefault(t, []).append((r, c))

    # Dilated per type, lowest type first, so a later type cannot eat an earlier
    # one where two clicks sit within a disk of each other.
    for t in sorted(clicks):
        pts = np.zeros(shape, bool)
        for r, c in clicks[t]:
            pts[r, c] = True
        if cfg.manual_seed_radius > 0:
            pts = morphology.dilation(pts, morphology.disk(cfg.manual_seed_radius))
        labels[pts & (labels == 0)] = t

    warnings = []
    if len(clicks) == 1 and 1 in clicks:
        warnings.append("Only tumor seeds planted; SRG needs competing types to stop.")
    return labels, {"status": "OK" if (labels == 1).any() else "NO SEED",
                    "core_area": int((labels == 1).sum()),
                    "n_clicks": len(points),
                    "types": {f"type {t}": int((labels == t).sum()) for t in sorted(clicks)},
                    "components": [], "warnings": warnings}
