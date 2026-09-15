"""
anatomy_exclusion.py — Phase 2a: orbit and skull-base exclusion (ablation).

Purpose : Remove bright non-tumor anatomy from candidacy before ranking.
Function : orbit_mask() flags bright, round components in the upper-anterior head
          by intensity + circularity + position; skull_base_band() forbids a narrow
          bright floor band; forbidden_region() unions the enabled exclusions.
Notes   : Evaluated on the full test set with no aggregate gain — orbits are dark on
          most slices and already dropped by Otsu, and pituitary tumors are
          intermixed with normal enhancement. Retained for ablation; off by default.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology


def orbit_mask(img, head, depth, cfg):
    """
    Detect orbits: bright, roughly circular, compact components sitting in the
    upper (anterior) part of the head. Returns a boolean mask to forbid.

    Signature used (all measured, none hard-coded to pixel positions):
      - intensity in the upper brightness range (orbits are bright on T1-CE),
      - high circularity (eyeballs are near-spherical -> round in-plane),
      - located in the upper fraction of the head bounding box,
      - bilateral / paired is a bonus but not required (works on one orbit too).
    """
    if not head.any():
        return np.zeros_like(head, bool)

    ys, xs = np.nonzero(head)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = max(y1 - y0, 1)

    # Bright tissue inside the head
    thr = np.percentile(img[head], cfg.orbit_bright_pct)
    bright = head & (img >= thr)
    bright = morphology.opening(bright, morphology.disk(1))

    forbidden = np.zeros_like(head, bool)
    lab, n = ndi.label(bright)
    for i in range(1, n + 1):
        comp = lab == i
        area = int(comp.sum())
        if area < cfg.orbit_min_area or area > cfg.orbit_max_area:
            continue
        props = measure.regionprops(comp.astype(int))[0]
        perim = props.perimeter if props.perimeter > 0 else 1.0
        circularity = 4 * np.pi * props.area / (perim * perim)
        cy = props.centroid[0]
        rel_y = (cy - y0) / h                      # 0 = top of head, 1 = bottom
        # Orbit = round AND in the upper region of the head
        if circularity >= cfg.orbit_min_circularity and rel_y <= cfg.orbit_max_rel_y:
            # Dilate slightly so the whole eyeball plus its rim is forbidden
            forbidden |= morphology.binary_dilation(comp, morphology.disk(2))
    return forbidden & head


def skull_base_band(head, depth, cfg):
    """
    Forbid a thin band along the very bottom of the head mask, where bright
    skull-base marrow traps some seeds on axial/coronal slices. Uses head
    geometry only (no intensity), kept narrow to avoid touching central tumors.
    """
    if not head.any():
        return np.zeros_like(head, bool)
    ys, xs = np.nonzero(head)
    y1 = ys.max(); h = max(y1 - ys.min(), 1)
    band = np.zeros_like(head, bool)
    cut = int(y1 - cfg.skullbase_band_frac * h)
    band[cut:, :] = True
    return band & head & (depth <= cfg.skullbase_depth)   # only shallow floor tissue


def forbidden_region(img, head, depth, cfg):
    """Union of the anatomical structures to exclude from candidacy."""
    forb = np.zeros_like(head, bool)
    if cfg.exclude_orbits:
        forb |= orbit_mask(img, head, depth, cfg)
    if cfg.exclude_skull_base:
        forb |= skull_base_band(head, depth, cfg)
    return forb
