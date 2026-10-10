"""
explain.py — Traceable computation of every evaluation metric for one slice.

Purpose : Back the GUI's Evaluation step: for the slice just segmented, show each
          metric's value together with the equation, the substituted numbers, and
          the result.
Function : explain(result, leak_ratio) returns one entry per metric in METRICS,
          each a dict with the value, a one-line verdict, the equation, and the
          worked steps. The pixel counts the metrics share (|M|, |G|, TP, FP, FN,
          TN) come from counts() / count_lines(), shown once and referenced. Values are recomputed
          here from the masks the run produced, using the same definitions as
          esrg.metrics (Chapter 3, Equations 3.32–3.39), so the panel and the
          batch evaluation agree to the last digit.
Notes   : Lines are kept short because the telemetry panel is narrow. Metrics that
          need a ground truth report that requirement instead of a value.
"""
import numpy as np
from scipy import ndimage as ndi

from . import metrics

# key, panel label, objective it evaluates
METRICS = [
    ("dsc", "DSC", "Overall delineation"),
    ("iou", "IoU", "Overall delineation"),
    ("recall", "Recall", "Objective 2 · undersegmentation"),
    ("precision", "Precision", "Objective 3 · boundary leakage"),
    ("leakage", "Leakage", "Objective 3 · boundary leakage"),
    ("hd95", "HD95", "Objective 3 · boundary accuracy"),
    ("assd", "ASSD", "Objective 3 · boundary accuracy"),
    ("seed", "Seed hit", "Objective 1 · seed selection"),
]
NEEDS_GT = {"dsc", "iou", "recall", "precision", "leakage", "hd95", "assd", "seed"}
SUCCESS_DSC = 0.70


def _n(v):
    """Integer with thousands separators."""
    return f"{int(v):,}"


def _f(v, d=4):
    return "undefined" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def _counts(pred, gt):
    tp, fp, fn, tn = metrics.confusion(pred, gt)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "M": int(pred.sum()), "G": int(gt.sum()), "px": int(pred.size)}


def counts(result):
    """The pixel counts the metrics share, or None without a ground truth."""
    if result.gt is None or result.mask is None:
        return None
    return _counts(result.mask.astype(bool), result.gt.astype(bool))


def count_lines(c):
    """M compared with G pixel by pixel; shown once, above every metric."""
    return [
        f"|M| = {_n(c['M'])} px,  |G| = {_n(c['G'])} px",
        f"TP = |M ∩ G| = {_n(c['tp'])}   (M = 1, G = 1)",
        f"FP = |M \\ G| = {_n(c['fp'])}   (M = 1, G = 0)",
        f"FN = |G \\ M| = {_n(c['fn'])}   (M = 0, G = 1)",
        f"TN = rest   = {_n(c['tn'])}   (not used)",
        f"check: TP + FP = {_n(c['tp'] + c['fp'])} = |M|",
        f"       TP + FN = {_n(c['tp'] + c['fn'])} = |G|",
    ]


def _ratio(num, den):
    return num / den if den else float("nan")


def _dsc(c):
    num, den = 2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]
    v = _ratio(num, den)
    verdict = ("undefined (both masks empty)" if not den else
               f"success (DSC ≥ {SUCCESS_DSC:.2f})" if v >= SUCCESS_DSC else
               f"below the {SUCCESS_DSC:.2f} success threshold")
    return v, verdict, ["DSC = 2TP / (2TP + FP + FN)   (Eq. 3.32)",
                        "    = 2|M ∩ G| / (|M| + |G|)"], [
        "Substitute the pixel counts:",
        f"DSC = 2({_n(c['tp'])}) / (2({_n(c['tp'])}) + {_n(c['fp'])} + {_n(c['fn'])})",
        f"    = {_n(num)} / {_n(den)}" if den else "    = 0 / 0",
        f"    = {_f(v)}",
        f"Success flag: {_f(v)} {'≥' if den and v >= SUCCESS_DSC else '<'} {SUCCESS_DSC:.2f}",
    ]


def _iou(c):
    den = c["tp"] + c["fp"] + c["fn"]
    v = _ratio(c["tp"], den)
    d = _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])
    return v, "overlap divided by union of M and G", [
        "IoU = TP / (TP + FP + FN)   (Eq. 3.33)",
        "    = |M ∩ G| / |M ∪ G|"], [
        "Substitute the pixel counts:",
        f"IoU = {_n(c['tp'])} / ({_n(c['tp'])} + {_n(c['fp'])} + {_n(c['fn'])})",
        f"    = {_n(c['tp'])} / {_n(den)}",
        f"    = {_f(v)}",
        f"Cross-check: IoU = DSC / (2 − DSC)",
        f"    = {_f(d)} / (2 − {_f(d)}) = {_f(d / (2 - d) if den else float('nan'))}",
    ]


def _recall(c):
    v = _ratio(c["tp"], c["tp"] + c["fn"])
    return v, f"{_f(100 * (1 - v), 1)}% of the tumor was missed", [
        "Recall = TP / (TP + FN)   (Eq. 3.35)",
        "       = |M ∩ G| / |G|",
        "Missed share = FN / |G| = 1 − Recall"], [
        "Substitute the pixel counts:",
        f"Recall = {_n(c['tp'])} / ({_n(c['tp'])} + {_n(c['fn'])})",
        f"       = {_n(c['tp'])} / {_n(c['tp'] + c['fn'])}",
        f"       = {_f(v)}",
        f"Missed = {_n(c['fn'])} / {_n(c['G'])} = {_f(1 - v)}",
    ]


def _precision(c):
    den = c["tp"] + c["fp"]
    v = _ratio(c["tp"], den)
    verdict = ("undefined: empty prediction" if not den else
               f"{_f(100 * (1 - v), 1)}% of the prediction is non-tumor")
    return v, verdict, [
        "Precision = TP / (TP + FP)   (Eq. 3.34)",
        "          = |M ∩ G| / |M|",
        "Spill share = FP / |M| = 1 − Precision"], [
        "Substitute the pixel counts:",
        f"Precision = {_n(c['tp'])} / ({_n(c['tp'])} + {_n(c['fp'])})",
        f"          = {_n(c['tp'])} / {_n(den)}" if den else "          = 0 / 0",
        f"          = {_f(v)}",
    ]


def _leakage(c, lam):
    r = _ratio(c["M"], c["G"])
    leaked = bool(c["G"] and r > lam)
    verdict = (f"LEAKED (|M| > {lam:g}|G|)" if leaked else
               "no leakage; region smaller than tumor" if r < 1 else
               f"no leakage (ratio ≤ {lam:g})")
    return r, verdict, [
        "Area ratio  ρ = |M| / |G|",
        f"Leaked  ⇔  ρ > λ,  λ = {lam:g}   (Eq. 3.39)",
        "Leakage Rate = share of slices",
        "with Leaked = 1 (batch level)"], [
        "Substitute the pixel counts:",
        f"ρ = {_n(c['M'])} / {_n(c['G'])} = {_f(r, 3)}",
        f"{_f(r, 3)} {'>' if leaked else '≤'} {lam:g}  →  Leaked = {int(leaked)}",
        f"FP pixels outside the tumor: {_n(c['fp'])}",
    ]


def _boundary_detail(pred, gt):
    bp, bg = metrics._boundary(pred), metrics._boundary(gt)
    if not bp.any() or not bg.any():
        return None
    d_pg = ndi.distance_transform_edt(~bg)[bp]
    d_gp = ndi.distance_transform_edt(~bp)[bg]
    return {"nb_p": int(bp.sum()), "nb_g": int(bg.sum()), "d_pg": d_pg, "d_gp": d_gp,
            "d": np.concatenate([d_pg, d_gp])}


def _dist_sources(c, b):
    return [
        "∂M = pixels of M with a 4-neighbour",
        "     outside M (Eq. 3.36)",
        f"|∂M| = {_n(b['nb_p'])} boundary px",
        f"|∂G| = {_n(b['nb_g'])} boundary px",
        "Step 1 · for each p ∈ ∂M: Euclidean",
        "  distance to nearest pixel of ∂G",
        f"  n = {_n(b['d_pg'].size)}: min {b['d_pg'].min():.2f}, "
        f"median {np.median(b['d_pg']):.2f},",
        f"  max {b['d_pg'].max():.2f} px",
        "Step 2 · for each q ∈ ∂G: distance",
        "  to nearest pixel of ∂M",
        f"  n = {_n(b['d_gp'].size)}: min {b['d_gp'].min():.2f}, "
        f"median {np.median(b['d_gp']):.2f},",
        f"  max {b['d_gp'].max():.2f} px",
        f"Step 3 · pool both sets: n = {_n(b['d'].size)}",
    ]


def _hd95(pred, gt, c):
    b = _boundary_detail(pred, gt)
    if b is None:
        diag = float(np.hypot(*gt.shape))
        return float("nan"), "undefined: a mask is empty", [
            "HD95 = P95 of pooled boundary",
            "       distances   (Eq. 3.37)"], [
            "∂M or ∂G is empty, so no distance exists.",
            f"Batch analysis uses the worst case,",
            f"the image diagonal = {diag:.2f} px."]
    d = np.sort(b["d"])
    n = d.size
    pos = 0.95 * (n - 1)
    i, frac = int(np.floor(pos)), pos - np.floor(pos)
    lo, hi = d[i], d[min(i + 1, n - 1)]
    v = float(np.percentile(b["d"], 95))
    return v, f"95% of boundary points lie within {v:.2f} px", [
        "HD95 = P95( {d(p,∂G): p∈∂M} ∪",
        "            {d(q,∂M): q∈∂G} )  (Eq. 3.37)",
        "P95 by linear interpolation of",
        "the sorted pooled distances"], _dist_sources(c, b) + [
        "Step 4 · sort ascending, d(1) … d(n)",
        f"  position = 0.95 × (n − 1)",
        f"           = 0.95 × {n - 1} = {pos:.2f}",
        f"  → between rank {i + 1} and {min(i + 2, n)}",
        f"  d({i + 1}) = {lo:.4f},  d({min(i + 2, n)}) = {hi:.4f}",
        "Step 5 · interpolate:",
        f"HD95 = d({i + 1}) + {frac:.2f} × (d({min(i + 2, n)}) − d({i + 1}))",
        f"     = {lo:.4f} + {frac:.2f} × {hi - lo:.4f}",
        f"     = {v:.4f} px",
        f"(Maximum = classic Hausdorff = {d[-1]:.2f} px)",
    ]


def _assd(pred, gt, c):
    b = _boundary_detail(pred, gt)
    if b is None:
        return float("nan"), "undefined: a mask is empty", [
            "ASSD = mean of pooled boundary distances"], [
            "∂M or ∂G is empty, so no distance exists."]
    s, n = float(b["d"].sum()), b["d"].size
    v = s / n
    return v, f"boundaries are {v:.2f} px apart on average", [
        "ASSD = ( Σ d(p,∂G) + Σ d(q,∂M) )",
        "       / ( |∂M| + |∂G| )"], _dist_sources(c, b) + [
        "Step 4 · average all pooled values:",
        f"Σ d(p,∂G) = {b['d_pg'].sum():.3f}",
        f"Σ d(q,∂M) = {b['d_gp'].sum():.3f}",
        f"ASSD = ({b['d_pg'].sum():.3f} + {b['d_gp'].sum():.3f})",
        f"       / ({_n(b['nb_p'])} + {_n(b['nb_g'])})",
        f"     = {s:.3f} / {_n(n)} = {v:.4f} px",
    ]


def _seed(result, gt):
    st = result.stage("seed")
    S = st.image.astype(bool) if st is not None else np.zeros_like(gt)
    mode = result.meta.get("seed_mode", "auto")
    origin = ("automatic: medial core of top-ranked candidate (Eq. 3.14)" if mode == "auto"
              else "manual clicks dilated to r = 3 disks" + (" and purified to their core"
                                                             if result.meta.get("method") == "esrg" else ""))
    if not S.any():
        return None, "no seed: no tumor candidate", [
            "Hit ⇔ S ⊆ G   (Eq. 3.38)"], [
            "Stage 5 produced no seed core, so the",
            "system reported 'no tumor candidate'.",
            "Counted in the batch as a miss."]
    inside = int((S & gt).sum())
    outside = int((S & ~gt).sum())
    hit = outside == 0
    r, cc = np.argwhere(S).mean(axis=0)
    dist = metrics.seed_distance(S, gt)
    return float(hit), ("HIT: whole seed core inside the tumor" if hit else
                        "MISS: part of the seed lies outside"), [
        "Hit ⇔ S ⊆ G  ⇔  |S \\ G| = 0   (Eq. 3.38)",
        "SHR = hits / seeded slices (batch)",
        "Inside share = |S ∩ G| / |S|",
        "Localization = distance from the",
        "  centroid of S to the nearest G px"], [
        "S = Stage 5 · Seed core:",
        f"  {origin}",
        f"|S|     = {_n(S.sum())} px",
        f"|S ∩ G| = {_n(inside)} px inside the tumor",
        f"|S \\ G| = {_n(outside)} px outside the tumor",
        f"Hit = [{_n(outside)} = 0] = {int(hit)}",
        f"Inside share = {_n(inside)} / {_n(S.sum())} = {inside / S.sum():.4f}",
        f"Centroid of S = (row {r:.1f}, col {cc:.1f})",
        f"Distance to nearest tumor px = {dist:.2f} px" + ("  (inside)" if dist == 0 else ""),
    ]


def _fit(lines, width=38):
    """Move a trailing '(Eq. …)' reference to its own line when a line is too wide."""
    out = []
    for line in lines:
        i = line.rfind("(Eq.")
        if len(line) > width and i > 0:
            out += [line[:i].rstrip(), " " * 6 + line[i:]]
        else:
            out.append(line)
    return out


def explain(result, leak_ratio=2.0):
    """{metric key: entry} for every metric in METRICS (see module docstring)."""
    out = {}
    gt = result.gt.astype(bool) if result.gt is not None else None
    pred = result.mask.astype(bool) if result.mask is not None else None
    c = _counts(pred, gt) if gt is not None else None
    for key, label, objective in METRICS:
        entry = {"key": key, "label": label, "objective": objective}
        if key in NEEDS_GT and gt is None:
            entry.update(value=None, verdict="requires a ground-truth mask",
                         formula=[], steps=[
                             "No ground-truth mask was found next to",
                             "this image (../masks/<name>.png), so",
                             "this metric cannot be computed."])
            out[key] = entry
            continue
        if key == "dsc":
            v, verdict, formula, steps = _dsc(c)
        elif key == "iou":
            v, verdict, formula, steps = _iou(c)
        elif key == "recall":
            v, verdict, formula, steps = _recall(c)
        elif key == "precision":
            v, verdict, formula, steps = _precision(c)
        elif key == "leakage":
            v, verdict, formula, steps = _leakage(c, leak_ratio)
        elif key == "hd95":
            v, verdict, formula, steps = _hd95(pred, gt, c)
        elif key == "assd":
            v, verdict, formula, steps = _assd(pred, gt, c)
        else:
            v, verdict, formula, steps = _seed(result, gt)
        entry.update(value=v, verdict=verdict, formula=_fit(formula), steps=_fit(steps))
        out[key] = entry
    return out


def value_text(entry):
    """Headline value of an entry, formatted for the panel."""
    k, v = entry["key"], entry["value"]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if k in ("hd95", "assd"):
        return f"{v:.2f} px"
    if k == "leakage":
        return f"ratio {v:.3f}"
    if k == "seed":
        return "HIT" if v else "MISS"
    return f"{v:.4f}"
