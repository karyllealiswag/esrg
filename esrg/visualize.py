"""
visualize.py — Rendering of stages and the annotated output.

Purpose : Convert stage arrays and results into displayable RGB for the GUI and for
          saved figures.
Function : render_stage() draws each stage appropriately (gray, heat, mask outline,
          overlay); overlay_result() fills the prediction and outlines ground truth,
          or colours TP/FP/FN separately; score_line() formats the caption metrics.
Notes   : Colours are fixed (red=prediction/FP, green=ground truth, orange=FN) so
          leakage and undersegmentation are visible at a glance.
"""
import numpy as np
from scipy import ndimage as ndi

RED = (239, 68, 68)        # prediction / false positive
GREEN = (34, 197, 94)      # ground truth
ORANGE = (249, 115, 22)    # false negative
BLUE = (59, 130, 246)      # candidates
YELLOW = (234, 179, 8)     # mask outline


def _gray_rgb(img):
    a = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([a] * 3, -1)


def _heat(arr, mask=None):
    """Simple blue-to-yellow ramp for the log-domain view."""
    v = arr.copy().astype(float)
    valid = mask if mask is not None else np.ones_like(v, bool)
    if valid.sum() == 0:
        return _gray_rgb(v)
    lo, hi = v[valid].min(), v[valid].max()
    t = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros(v.shape + (3,), np.uint8)
    rgb[..., 0] = (255 * t).astype(np.uint8)
    rgb[..., 1] = (255 * t ** 0.7).astype(np.uint8)
    rgb[..., 2] = (255 * (1 - t)).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = 0
    return rgb


def outline(mask, width=1):
    """Boundary pixels of a binary mask."""
    m = mask.astype(bool)
    er = ndi.binary_erosion(m, iterations=max(1, width))
    return m & ~er


def render_stage(stage, base_img, gt=None, opacity=0.55):
    """RGB view of one pipeline stage, drawn over the input slice."""
    rgb = _gray_rgb(base_img)

    if stage.kind == "gray":
        out = _gray_rgb(stage.image)
    elif stage.kind == "heat":
        out = _heat(stage.image, stage.image > stage.image.min())
    elif stage.kind == "mask":
        out = rgb.copy()
        m = stage.image.astype(bool)
        if stage.key == "mask":                       # head mask: outline only
            out[outline(m, 2)] = YELLOW
        elif stage.key == "candidates":
            out[m] = (np.array(BLUE) * opacity + out[m] * (1 - opacity)).astype(np.uint8)
        elif stage.key == "seed":
            if m.any():
                out[ndi.binary_dilation(m, iterations=1)] = RED
        else:                                          # growth
            out[m] = (np.array(RED) * opacity + out[m] * (1 - opacity)).astype(np.uint8)
    else:
        out = overlay_result(base_img, stage.image, gt, opacity)

    if gt is not None and stage.kind in ("mask", "overlay") and stage.key != "seed":
        out[outline(gt)] = GREEN
    return out


def overlay_result(base_img, pred, gt=None, opacity=0.55, error_mode=False):
    """
    Final annotated output.
      error_mode False: prediction filled red, ground truth as green contour
      error_mode True : true positive green, false positive red, false negative orange
    """
    out = _gray_rgb(base_img)
    pred = pred.astype(bool)

    if error_mode and gt is not None:
        gt = gt.astype(bool)
        for region, color in ((pred & gt, GREEN), (pred & ~gt, RED), (~pred & gt, ORANGE)):
            if region.any():
                out[region] = (np.array(color) * opacity
                               + out[region] * (1 - opacity)).astype(np.uint8)
    else:
        if pred.any():
            out[pred] = (np.array(RED) * opacity + out[pred] * (1 - opacity)).astype(np.uint8)
            out[outline(pred)] = RED
        if gt is not None and gt.any():
            out[outline(gt.astype(bool))] = GREEN
    return out


def score_line(scores):
    """One-line caption for the annotated output."""
    if not scores:
        return "No ground truth available — scores not computed."
    def f(k):
        v = scores.get(k)
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    return (f"DSC {f('dsc')}   IoU {f('iou')}   Precision {f('precision')}   "
            f"Recall {f('recall')}   HD95 {f('hd95')}   {scores.get('seconds', 0):.2f}s")
