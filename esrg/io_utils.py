"""
io_utils.py — Image and mask I/O plus BRISC filename parsing.

Purpose : Load MRI slices and ground-truth masks, and decode BRISC metadata
          (tumor type, plane) from filenames.
Function : Images load as grayscale float64 in [0,255] (downscaled past max_side);
          masks load as boolean via a >127 threshold with nearest-neighbour resize
          to match the image; a regex maps BRISC names to tumor/plane fields.
Notes   : BRISC masks are effectively binary with a thin anti-aliased edge fringe,
          so the >127 threshold is stable. Source files are never modified.
"""
import os
import re
import numpy as np
from PIL import Image

TUMOR_CODES = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "nt": "no_tumor"}
PLANE_CODES = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
_NAME_RE = re.compile(r"brisc2025_(train|test)_(\d+)_([a-z]{2})_([a-z]{2})_t1", re.I)


def load_image(path, max_side=512):
    """Read any image as grayscale float64 in [0, 255]; downscale if larger than max_side."""
    img = Image.open(path).convert("L")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64), scale


def load_mask(path, shape):
    """Read a binary mask (>127 = tumor) resized with nearest-neighbour to match `shape`."""
    m = Image.open(path).convert("L")
    if m.size != (shape[1], shape[0]):
        m = m.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 127


def find_mask_for(image_path):
    # Locate the BRISC mask paired with an image by filename convention.
    """Locate the BRISC mask for an image: ../masks/<stem>.png, or <stem>.png next to it."""
    folder, fname = os.path.split(image_path)
    stem = os.path.splitext(fname)[0]
    candidates = [
        os.path.join(os.path.dirname(folder), "masks", stem + ".png"),
        os.path.join(folder, stem + ".png"),
        os.path.join(folder, stem + "_mask.png"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.path.abspath(c) != os.path.abspath(image_path):
            return c
    return None


def parse_brisc_name(path):
    """Return {'split','index','tumor','plane'} from a BRISC filename (unknown -> None)."""
    m = _NAME_RE.search(os.path.basename(path))
    if not m:
        return {"split": None, "index": None, "tumor": None, "plane": None}
    split, idx, t, p = m.groups()
    return {"split": split, "index": int(idx),
            "tumor": TUMOR_CODES.get(t.lower(), t), "plane": PLANE_CODES.get(p.lower(), p)}


def list_pairs(root):
    # Enumerate (image, mask) pairs in a split's images/ + masks/ folders.
    """List (image_path, mask_path) pairs in a BRISC split folder containing images/ and masks/."""
    img_dir = os.path.join(root, "images")
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            ip = os.path.join(img_dir, f)
            pairs.append((ip, find_mask_for(ip)))
    return pairs


def save_png(arr, path):
    """Save uint8/bool/float [0,255] arrays (gray or RGB) as PNG."""
    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(a).save(path)
