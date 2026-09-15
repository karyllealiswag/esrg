"""
preprocessing.py — Phase A: normalization, head masking, N4, log transform.

Purpose : Prepare a slice for seeding and growth, identically for ESRG and the
          SRG baseline.
Function : normalize() rescales to [0,255] on robust percentiles; head_mask()
          isolates the head (not the brain) via Otsu + granulometry and returns a
          depth map; log_transform() sends intensities to the log domain so the
          multiplicative bias field becomes additive (Objective 2); noise_floor()
          gives a robust sigma for the stopping bound (Objective 3).
Notes   : Head masking is deliberate — on 2D coronal/sagittal slices the brain
          cannot be separated from face/neck, so interior filtering is delegated to
          seed selection. N4 is an ablation arm only, never part of ESRG.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure


# ─── 1. Normalization ─────────────────────────────────────────────────────────
def normalize(img):
    """Linear rescale to [0, 255] using the 0.5 / 99.5 percentiles (robust to hot pixels)."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float64)
    return np.clip((img - lo) / (hi - lo), 0, 1) * 255.0


# ─── 2. Skull stripping ───────────────────────────────────────────────────────
def _largest_cc(mask):
    """Keep the largest connected component of a binary mask."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask.copy()
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _image_border_band(shape, width):
    """Boolean band of `width` px along the image border (neck/face touch it, scalp mostly not)."""
    b = np.zeros(shape, bool)
    b[:width, :] = b[-width:, :] = b[:, :width] = b[:, -width:] = True
    return b


def head_mask(img, cfg):
    """
    Returns (mask, depth, info).

    This isolates the HEAD, not the brain. On 2D T1 slices the face and neck are
    thick structures joined to the brain through the skull base, so no erosion
    radius separates them on coronal or sagittal views. Rather than pretend
    otherwise, this stage bounds the search region and hands the interior test to
    seed selection, which excludes rim-adjacent components (Objective 1).

      mask  : head region, holes filled
      depth : distance of each pixel from the head outline; the seed stage uses
              it to tell interior tissue from scalp
      info  : granulometry diagnostics, surfaced in the GUI's Stage 2 panel
    """
    info = {"radius": None, "area_drop": 0.0, "ring_found": False, "warnings": []}
    g = filters.gaussian(img, sigma=1.0, preserve_range=True)

    # Tissue vs background (air and cortical bone are both dark on T1)
    tissue = g > filters.threshold_otsu(g)

    # Head = largest blob after removing thin frames / arrows / text, holes filled
    opened = morphology.opening(tissue, morphology.disk(cfg.frame_open_radius))
    head = ndi.binary_fill_holes(_largest_cc(opened))
    if head.sum() < 0.05 * head.size:
        info["warnings"].append("Head region very small; check the input image.")
        return head, ndi.distance_transform_edt(head), info

    depth = ndi.distance_transform_edt(head)
    head_area = float(head.sum())

    # Granulometry, reported only as a diagnostic: the radius of the sharpest
    # area loss marks the skull ring where one exists (mostly axial slices).
    # This does not affect the returned mask or depth; it only fills `info`.
    deep = depth >= 0.5 * depth.max()
    tissue_head = tissue & head
    prev = float(tissue_head.sum())
    best_r, best_drop = None, 0.0
    for r in range(1, cfg.ss_max_radius + 1):
        lab, n = ndi.label(morphology.erosion(tissue_head, morphology.disk(r)))
        if n == 0:
            break
        overlap = ndi.sum(deep, lab, range(1, n + 1))
        if overlap.max() == 0:
            break
        # Winning component's area via one bincount pass (identical to a mask sum).
        a = float(np.bincount(lab.ravel())[int(np.argmax(overlap)) + 1])
        if a < cfg.ss_min_brain_frac * head_area:
            break
        if prev - a > best_drop:
            best_drop, best_r = prev - a, r
        prev = a

    info["radius"] = int(best_r) if best_r else None
    info["area_drop"] = float(best_drop / head_area)
    info["ring_found"] = bool(info["area_drop"] >= 0.02)
    info["head_frac_of_image"] = float(head_area / head.size)
    if not info["ring_found"]:
        info["warnings"].append(
            "No distinct skull ring; head mask retains scalp and facial tissue. "
            "Interior filtering is handled by seed selection.")
    return head, depth, info


# ─── 3. N4 bias correction (ablation only) ────────────────────────────────────
def n4_correct(img, mask, shrink=2, iterations=(50, 50, 30)):
    """N4ITK via SimpleITK. Returns (corrected, bias_field). Raises ImportError if unavailable."""
    import SimpleITK as sitk
    src = sitk.GetImageFromArray(img.astype(np.float32) + 1.0)      # N4 needs positive values
    msk = sitk.GetImageFromArray(mask.astype(np.uint8))
    small = sitk.Shrink(src, [shrink] * 2)
    small_m = sitk.Shrink(msk, [shrink] * 2)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(list(iterations))
    n4.Execute(small, small_m)
    log_bias = n4.GetLogBiasFieldAsImage(src)                        # full-resolution field
    bias = np.exp(sitk.GetArrayFromImage(log_bias)).astype(np.float64)
    corrected = (img + 1.0) / bias - 1.0
    corrected[~mask] = 0
    return np.clip(corrected, 0, None), bias


# ─── 4–5. Log domain + noise floor ────────────────────────────────────────────
def log_transform(img, eps=1.0):
    """L(x) = ln(I(x) + eps). Multiplicative bias becomes additive (Objective 2)."""
    return np.log(img + eps)


def noise_floor(L, mask, min_value=0.02):
    """Robust noise sigma of L in the brain: 1.4826 * MAD of (L - median3x3(L))."""
    resid = (L - ndi.median_filter(L, size=3))[mask]
    if resid.size == 0:
        return float(min_value)
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(max(1.4826 * mad, min_value))
