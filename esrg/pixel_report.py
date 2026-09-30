"""
pixel_report.py — Seed-pixel coordinates and grayscale values for the GUI.

Purpose : Report exactly which pixels seed the region growth and what grayscale
          they hold, so the growth arithmetic can be traced by hand.
Function : manual_groups() groups user clicks by seed region, auto_groups() lists
          the system-selected seed core, and describe() attaches the raw and
          normalized (0-255) value of every pixel plus the mean of each group;
          log_domain() gives the Stage 3 value L(x) of the same pixels.
Notes   : Pure NumPy, no GUI. Coordinates are (row, col), 0-based, in the working
          image (after any downscale to max_side). "raw" is the file value; "norm"
          is preprocessing.normalize(raw), the value the pipeline actually grows on.
"""
import numpy as np


def manual_groups(points):
    """
    points are (r, c, type) user clicks. Returns [(label, [(r, c), ...])] with one
    group per seed type in ascending order. Repeated clicks on one pixel collapse
    to a single entry (the seed map is boolean, so the pipeline sees them once).
    """
    by_type = {}
    for p in points:
        t = int(p[2]) if len(p) > 2 else 1
        pix = by_type.setdefault(t, [])
        rc = (int(p[0]), int(p[1]))
        if rc not in pix:
            pix.append(rc)
    return [(f"Region {t}" + (" (tumor)" if t == 1 else ""), by_type[t])
            for t in sorted(by_type)]


def auto_groups(core):
    """core is the boolean seed mask chosen by the system; [] when it is empty."""
    px = [(int(r), int(c)) for r, c in np.argwhere(core)]
    return [("System seed core", px)] if px else []


def describe(groups, raw, norm):
    """
    Attach grayscale values to each group. Returns one dict per group:
    label, n, pixels [{row, col, raw, norm}], mean_raw, mean_norm.
    """
    out = []
    for label, pix in groups:
        rows = [{"row": r, "col": c, "raw": float(raw[r, c]), "norm": float(norm[r, c])}
                for r, c in pix]
        out.append({"label": label, "n": len(rows), "pixels": rows,
                    "mean_raw": float(np.mean([p["raw"] for p in rows])),
                    "mean_norm": float(np.mean([p["norm"] for p in rows]))})
    return out


def log_domain(groups, I, L, mask, eps, use_log):
    """
    Log-domain value of every seed pixel, read from the arrays the pipeline used.
    I is the head-masked normalized slice (0-255) and L the Stage 3 output, so
    L(x) = ln(I(x) + eps) when use_log, else L(x) = I(x). Returns one dict per
    group: label, n, pixels [{row, col, I, L, in_head}], mean_L.
    """
    out = []
    for label, pix in groups:
        rows = [{"row": r, "col": c, "I": float(I[r, c]), "L": float(L[r, c]),
                 "in_head": bool(mask[r, c])} for r, c in pix]
        out.append({"label": label, "n": len(rows), "pixels": rows,
                    "mean_L": float(np.mean([p["L"] for p in rows]))})
    return out
