"""
metrics.py — Segmentation scores (Chapter 3, Section 3.4.4).

Purpose : Quantify a predicted mask against ground truth.
Function : evaluate() returns DSC, IoU, precision, recall (overlap) plus HD95 and
          ASSD (boundary) and a leakage flag; seed_hit() checks the seed core lies
          inside the tumor; summarize() aggregates per-metric mean/SD/median/IQR.
Notes   : NaN where a metric is undefined (empty mask). Boundary distances use the
          symmetric surface distance; HD95 is the 95th percentile to resist outliers.
"""
import numpy as np
from scipy import ndimage as ndi


def confusion(pred, gt):
    # True/false positive/negative pixel counts for one mask pair.
    pred, gt = pred.astype(bool), gt.astype(bool)
    return (int((pred & gt).sum()), int((pred & ~gt).sum()),
            int((~pred & gt).sum()), int((~pred & ~gt).sum()))


def _boundary(mask):
    return mask & ~ndi.binary_erosion(mask, ndi.generate_binary_structure(2, 1))


def surface_distances(pred, gt):
    # Symmetric boundary-to-boundary distances (feeds HD95 and ASSD).
    """Symmetric boundary-to-boundary distances in pixels; empty when either mask is empty."""
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() or not bg.any():
        return np.array([])
    d_to_gt = ndi.distance_transform_edt(~bg)
    d_to_pr = ndi.distance_transform_edt(~bp)
    return np.concatenate([d_to_gt[bp], d_to_pr[bg]])


def evaluate(pred, gt, cfg=None):
    """All scores for one image. NaN where a metric is undefined."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp, fp, fn, tn = confusion(pred, gt)

    m = {"tp": tp, "fp": fp, "fn": fn,
         "pred_area": int(pred.sum()), "gt_area": int(gt.sum())}
    m["dsc"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    m["iou"] = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    m["precision"] = tp / (tp + fp) if (tp + fp) else np.nan
    m["recall"] = tp / (tp + fn) if (tp + fn) else np.nan

    d = surface_distances(pred, gt)
    m["hd95"] = float(np.percentile(d, 95)) if d.size else np.nan
    m["assd"] = float(d.mean()) if d.size else np.nan

    ratio = m["pred_area"] / m["gt_area"] if m["gt_area"] else np.nan
    m["area_ratio"] = ratio
    m["leaked"] = bool(ratio > (cfg.leak_ratio if cfg else 2.0)) if m["gt_area"] else False
    return m


def seed_hit(core, gt):
    """Objective 1: True when the whole seed core lies inside the ground truth."""
    if not core.any():
        return None
    return bool((core & ~gt.astype(bool)).sum() == 0)


def summarize(rows, keys=("dsc", "iou", "precision", "recall", "hd95", "seconds")):
    """Mean/SD/median/IQR per metric, ignoring NaN."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None], float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        out[k] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "median": float(np.median(v)),
                  "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}
    return out
