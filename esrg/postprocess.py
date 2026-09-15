"""
postprocess.py — Phase D: mask cleanup.

Purpose : Turn the grown region into the final tumor mask.
Function : Fills interior holes (necrotic cores of ring-enhancing tumors), applies a
          light morphological opening (never one that empties the mask), and keeps
          only the connected component containing the seed core.
Notes   : Records before/after areas so the GUI can show each cleanup step's effect.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology


def postprocess(region, core, cfg):
    """Fill necrotic cores, shave thin protrusions, keep the component holding the seed."""
    info = {"before": int(region.sum())}
    out = ndi.binary_fill_holes(region)
    info["after_fill"] = int(out.sum())

    if cfg.post_open_radius > 0:
        opened = morphology.opening(out, morphology.disk(cfg.post_open_radius))
        if opened.any():                       # never open the region out of existence
            out = opened
    info["after_open"] = int(out.sum())

    lab, n = ndi.label(out)
    if n > 1 and core.any():
        ids = set(lab[core & (lab > 0)].tolist())
        out = np.isin(lab, list(ids)) if ids else out
    info["after"] = int(out.sum())
    info["components_dropped"] = max(0, n - 1)
    return out, info
