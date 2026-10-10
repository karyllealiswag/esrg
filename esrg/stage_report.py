"""
stage_report.py — End-to-end, stage-by-stage computation report for the GUI.

Purpose : Let a reviewer verify, for the slice just run, that every stage does what
          Chapter 3 states: what enters the stage and from where, the equation and
          parameters applied, the computed values, and what leaves for which stage.
Function : report(result, cfg, key, metrics) returns a list of display blocks
          (kind, payload) for one stage tab, starting with its title. Repeated
          per-pixel work is shown as the equation once, then a table of inputs and
          results; whole-image work shows the image-level quantities. Values are
          read from the arrays and traces the run itself produced
          (pipeline.run(record=True)), never re-derived by a second implementation.
Notes   : Pure NumPy, no GUI. Block kinds: head, sub, input, output, eq, kv, table,
          text, note, good, bad, anchor, tips. Explanations are not printed: a "tips"
          block maps terms that appear in the stage's text to the explanation the GUI
          shows as a tooltip when the pointer rests on them. An "anchor" block marks a
          place the GUI can scroll to (the OBJ buttons). Tables are kept within ~66
          characters so they fit the telemetry panel without wrapping.
"""
import os

import numpy as np

from . import explain as evalx
from . import pixel_report

PIXEL_CAP = 50          # rows per pass in the growth pixel table
SEED_CAP = 40           # rows in seed-pixel tables


# ─── formatting ───────────────────────────────────────────────────────────────
def _f(v, d=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}"


def _n(v):
    return f"{int(v):,}"


def _arr(a, kind="gray"):
    """One-line summary of an array handed between stages."""
    h, w = a.shape
    if kind == "mask":
        return f"{w}×{h} binary, {_n(a.sum())} px set"
    return f"{w}×{h}, min {float(a.min()):.3f}, max {float(a.max()):.3f}"


def _name(path):
    return os.path.basename(str(path or "")).replace("\\", "/").split("/")[-1]


class _B:
    """Block builder."""

    def __init__(self):
        self.out = []

    def __call__(self, kind, payload):
        self.out.append((kind, payload))
        return self

    def table(self, cols, rows, note=None, align=None):
        self.out.append(("table", {"cols": cols, "rows": rows, "note": note, "align": align}))


# ─── shared pieces ────────────────────────────────────────────────────────────
def _seed_pixels(result):
    seed = result.stage("seed")
    if seed is None or not seed.image.any():
        return []
    return [(int(r), int(c)) for r, c in np.argwhere(seed.image)]


def _seed_note(shown, total):
    return None if shown >= total else f"First {shown} of |S| = {_n(total)} seed-core pixels."


def _stage_name(result, key, fallback=""):
    st = result.stage(key)
    return st.name if st is not None else fallback


def _srg_manual(result):
    return result.meta.get("method") == "srg" and result.meta.get("seed_mode") == "manual"


def _working_image(result):
    """The image seeding and plain SRG read: I·H, or I itself for SRG manual (no head mask)."""
    I = result.stage("input").image
    mask = result.stage("mask")
    return np.where(mask.image, I, 0.0) if mask is not None else I


def _growth_image(result):
    """(array, symbol, stage) growth reads: L from Stage 3 when listed, else the working image."""
    log = result.stage("log")
    if log is not None:
        return log.image, ("L" if log.info.get("_use_log") else "I"), log.name
    return _working_image(result), "I", _stage_name(result, "input")


# Tooltip texts shared by several stages.
TIP_MU_A = ("μ_A: mean of the region's values. At the start the region A is the seed core S, "
            "so μ_A is the mean over S; ESRG then freezes it for each pass.")
TIP_S_A = ("s_A: sample standard deviation of the region's values (the seed core at the start). "
           "Stage 6 uses σ_A = max(s_A, σ_floor), which sets the bounds T_L = k_L·σ_A and T_G = k_G·σ_A.")
TIP_SIGMA_FLOOR = ("σ_floor: the smallest σ_A allowed. A uniform seed core can have s_A ≈ 0, which would "
                   "make T_L ≈ 0 and stop growth at once; the image's own noise level is used instead.")


# ─── Stage 1 · Input ──────────────────────────────────────────────────────────
def _input(result, cfg, b):
    st = result.stage("input")
    info, I = st.info, st.image
    nrm = info.get("_norm", {})
    lo, hi = nrm.get("lo"), nrm.get("hi")
    raw = info.get("_raw")
    b("sub", "INPUT")
    b("input", [("I₀", f"{_name(info.get('file'))}", "MRI slice from disk (grayscale)"),
                ("size", f"{info.get('size')}", f"longer side ≤ {cfg.max_side} px, scale ×{info.get('scale factor')}")])
    b("sub", "PROCESS · intensity normalization (Section 3.2.1, step 1b)")
    b("eq", ["I(x) = clip( (I₀(x) − p₀.₅) / (p₉₉.₅ − p₀.₅), 0, 1 ) × 255"])
    b("kv", [("p₀.₅  (0.5th percentile of I₀)", _f(lo, 3)), ("p₉₉.₅ (99.5th percentile of I₀)", _f(hi, 3)),
             ("scale 255/(p₉₉.₅ − p₀.₅)", _f(255.0 / (hi - lo), 6) if lo is not None and hi - lo > 1e-6 else "—")])
    if raw is not None and lo is not None:
        clipped_lo, clipped_hi = int((raw <= lo).sum()), int((raw >= hi).sum())
        b("note", f"{_n(clipped_lo)} px clipped to 0 and {_n(clipped_hi)} px clipped to 255 "
                  f"(of {_n(raw.size)}).")
        px = _seed_pixels(result)
        if px:
            b("sub", "COMPUTATION · seed-core pixels S (from Stage 5)")
            rows = []
            for r, c in px[:SEED_CAP]:
                v = float(raw[r, c])
                calc = min(max((v - lo) / (hi - lo), 0.0), 1.0) * 255.0
                rows.append([r, c, _f(v, 1), _f(calc, 3), _f(float(I[r, c]), 3)])
            b.table(["row", "col", "I₀(x)", "computed I(x)", "pipeline I(x)"], rows,
                    _seed_note(min(len(px), SEED_CAP), len(px)))
    b("sub", "OUTPUT")
    nxt = result.stages[1].name if len(result.stages) > 1 else "Evaluation"
    b("output", [("I", _arr(I), nxt)])
    b("tips", {
        "seed-core pixels S": "The pixels of the seed core chosen in Stage 5. They are listed here because "
                              "their values become the region's starting statistics μ_A and s_A in Stage 6.",
        "computed I(x)": "I(x) worked out from the equation above with this pixel's I₀(x).",
        "pipeline I(x)": "The value the pipeline itself produced; it must equal 'computed I(x)'.",
        "p₀.₅": "Percentiles instead of min/max, so a few hot or dead pixels cannot set the contrast range.",
    })


# ─── Stage 2 · Head mask ──────────────────────────────────────────────────────
GRANULOMETRY_WARNING = "No distinct skull ring"


def _mask(result, cfg, b):
    st = result.stage("mask")
    I, H = result.stage("input").image, st.image
    s = st.info.get("_steps", {})
    b("sub", "INPUT")
    b("input", [("I", _arr(I), "1 · Input")])
    b("sub", "PROCESS · head isolation (Section 3.2.1, step 1c)")
    b("eq", ["g = Gaussian_σ=1(I)",
             "tissue = g > t_Otsu(g)",
             f"H = fill_holes( largest_cc( open(tissue, disk {cfg.frame_open_radius}) ) )",
             "depth(x) = distance from x to the outline of H"])
    b("sub", "COMPUTATION · whole image")
    if s:
        b.table(["step", "result"], [
            ["Otsu threshold t of g", _f(s["otsu_t"], 3)],
            ["tissue = g > t", f"{_n(s['tissue_px'])} px"],
            [f"opening, disk r = {cfg.frame_open_radius}", f"{_n(s['opened_px'])} px"],
            ["largest connected component", f"{_n(s['largest_px'])} px"],
            ["holes filled → H", f"{_n(s['head_px'])} px ({s['head_px'] / s['image_px']:.1%} of image)"],
            ["max depth in H", f"{_f(s.get('depth_max'), 1)} px"]])
    # The skull-ring message comes from a diagnostic that does not change H.
    for w in st.info.get("warnings", []):
        if not w.startswith(GRANULOMETRY_WARNING):
            b("bad", w)
    b("sub", "OUTPUT")
    method = result.meta.get("method")
    used_by = ("3 · Log (I·H), 4 · Candidates, 6 · Growth (never leaves H), 7 · Final" if method == "esrg"
               else "4 · Candidates, 6 · Growth (background seeds and growth inside H), 7 · Final")
    b("output", [("H", _arr(H, "mask"), used_by),
                 ("depth", f"max {_f(s.get('depth_max'), 1)} px", "4 · Candidates (interior rule, Eq. 3.2)"),
                 ("I·H", "I with every x ∉ H set to 0", "3 · Log domain" if result.stage("log") else "4 · Candidates")])
    b("tips", {
        "t_Otsu": "Otsu's threshold: the gray level that best splits the histogram into two classes "
                  "(dark air/bone vs. tissue).",
        "open(": "Opening = erosion then dilation; removes thin frames, text and arrows before the "
                 "largest blob is taken as the head.",
        "depth(x)": "Euclidean distance from x to the nearest pixel outside H. Stage 4 uses it to tell "
                    "deep tissue from scalp.",
        "max depth in H": "depth_max: the depth of the most central head pixel.",
    })


# ─── Stage 2b · N4 (ablation) ─────────────────────────────────────────────────
def _n4(result, cfg, b):
    st = result.stage("n4")
    b("sub", "PROCESS · N4ITK bias correction (ablation arm only, not part of ESRG)")
    b("eq", ["I_N4(x) = (I(x) + 1) / b̂(x) − 1,  then renormalized to [0, 255]"])
    b("kv", [(k, str(v)) for k, v in st.info.items() if not k.startswith("_")])
    b("sub", "OUTPUT")
    b("output", [("I_N4", _arr(st.image), "3 · Log domain, replaces I·H")])


# ─── Stage 3 · Log domain + noise floor ───────────────────────────────────────
def _log(result, cfg, b):
    st = result.stage("log")
    srg = result.meta.get("method") == "srg"
    region = st.info.get("_region")
    if region is None:
        region = result.stage("mask").image
    src = st.info.get("_base")
    if src is None:
        src = np.where(region, (result.stage("n4") or result.stage("input")).image, 0.0)
    L, use_log, eps, nf = st.image, st.info["_use_log"], st.info["_eps"], st.info["_noise_floor"]
    whole = bool(region.all())
    where = "every pixel of the image" if whole else "x ∈ H"
    sym = "L" if use_log else "I"
    b("sub", "INPUT")
    if whole:
        b("input", [("I", _arr(src), "1 · Input (SRG grows on the whole image)")])
    else:
        b("input", [("I·H", _arr(src), "1 · Input × 2 · Head mask"), ("H", _arr(region, "mask"), "2 · Head mask")])
    if use_log or not srg:
        b("sub", "PROCESS · log transform (Eq. 3.6)" if use_log else "PROCESS · log transform OFF (ablation)")
        b("eq", [f"L(x) = ln( I(x) + ε ),  ε = {eps:g}" if use_log else "L(x) = I(x)   (pixels pass through)"])
        px = _seed_pixels(result)
        if px:
            g = pixel_report.log_domain([("seed", px[:SEED_CAP])], src, L, region, eps, use_log)[0]
            b("sub", "COMPUTATION · seed-core pixels S (from Stage 5)")
            b.table(["row", "col", "I(x)", "I(x)+ε", "L(x)"],
                    [[p["row"], p["col"], _f(p["I"], 4), _f(p["I"] + (eps if use_log else 0), 4), _f(p["L"], 6)]
                     for p in g["pixels"]], _seed_note(len(g["pixels"]), len(px)))
    b("sub", "PROCESS · noise floor σ_floor (Section 3.2.1, step 1e)")
    b("eq", [f"r(x) = {sym}(x) − median₃ₓ₃({sym})(x),   {where}",
             "MAD = median | r − median(r) |",
             f"σ_floor = max( 1.4826 · MAD, {nf['min_value']:g} )"])
    b("sub", "COMPUTATION · whole image" if whole else "COMPUTATION · whole head")
    b.table(["quantity", "value"], [
        ["pixels used", _n(nf["n"])], ["median(r)", _f(nf["median_resid"], 6)],
        ["MAD", _f(nf["mad"], 6)], ["1.4826 · MAD", _f(nf["raw_sigma"], 6)],
        ["σ_floor", _f(nf["sigma_floor"], 6)]])
    b("sub", "OUTPUT")
    growth = "6 · Region growing"
    if srg:
        out = []
        if use_log:
            out.append(("L", _arr(L), f"{growth} (region 1: δ, μ_loc)"))
        if cfg.srg_use_stopping:
            out.append(("σ_floor", _f(nf["sigma_floor"], 6), f"{growth} (lower bound of σ_A)"))
        b("output", out)
    else:
        b("output", [("L", _arr(L), f"{growth} (δ, μ_A, s_A)"),
                     ("σ_floor", _f(nf["sigma_floor"], 6), f"{growth} (lower bound of σ_A, Eq. 3.9)")])
    tips = {
        "r(x)": "Residual: how much a pixel differs from the median of its 3×3 neighbourhood. The median "
                "follows anatomy, so what is left is mostly noise.",
        "median(r)": f"The middle value of all {_n(nf['n'])} residuals r(x).",
        "MAD": "Median absolute deviation: the median of |r(x) − median(r)| over the same pixels. A spread "
               "measure that a few large residuals (edges) cannot inflate.",
        "1.4826": "1.4826 = 1/Φ⁻¹(0.75) turns the MAD into an estimate of the standard deviation of "
                  "Gaussian noise.",
        "σ_floor": TIP_SIGMA_FLOOR,
        growth: "L goes to region growing only. Stages 4–5 pick the seed on the normalized intensity I: "
                "enhancing tumor is defined as the bright top class of I, Otsu thresholds are not invariant "
                "to the log, and keeping seeding on I means switching the log changes growth alone. "
                "Growth then reads L at the seed positions, so μ_A and s_A are in log units.",
    }
    if use_log:
        tips["ε"] = "ε keeps ln defined at I = 0 (background) and limits how much the log stretches dark values."
    b("tips", tips)


# ─── Stage 4 · Candidates ─────────────────────────────────────────────────────
def _candidates(result, cfg, b):
    st = result.stage("candidates")
    si = st.info.get("_s_info", {})
    mode = result.meta.get("seed_mode")
    if mode != "auto":
        b("sub", "PROCESS · seeds supplied by the operator")
        disk = si.get("_click_disk")
        b("eq", [f"click disk = clicked pixels dilated by a disk of radius {cfg.manual_seed_radius}"])
        b("kv", [("clicks", str(si.get("n_clicks", "—"))),
                 ("click-disk area", f"{_n(disk.sum())} px" if disk is not None else "—"),
                 ("seed types", str(si.get("types", "—")))])
        b("note", "Automatic candidate selection (Eq. 3.1–3.3) is skipped in manual seeding.")
        b("sub", "OUTPUT")
        b("output", [("click disk", _arr(st.image, "mask"), "5 · Seed core")])
        return
    I = result.stage("input").image
    it = si.get("_interior", {})
    th = si.get("thresholds") or []
    b("sub", "INPUT")
    b("input", [("I·H", _arr(I), "1 · Input × 2 · Head mask"),
                ("depth", f"max {_f(it.get('depth_max'), 1)} px", "2 · Head mask")])
    b("sub", "PROCESS · interior region and multi-level Otsu (Eq. 3.1)")
    b("eq", ["interior = { x ∈ H : depth(x) ≥ max( c, η · depth_max ) }",
             "(t₁ … t_K−1) = argmax Σₖ ωₖ (μₖ − μ_T)²   over the interior histogram",
             "C = open( interior ∧ I ≥ t_K−1 , disk 1 )"])
    if it:
        b.table(["quantity", "value"], [
            ["c, η, K", f"{cfg.min_clearance:g} px, {cfg.interior_frac:g}, {cfg.otsu_classes}"],
            ["depth cut max(c, η·depth_max)", f"max({cfg.min_clearance:g}, {cfg.interior_frac:g}×{_f(it['depth_max'], 1)}) = {_f(it['depth_cut'], 2)} px"],
            ["interior pixels", f"{_n(it['px'])} of {_n(it['head_px'])} in H" + (" (fallback: all H)" if it.get("fallback") else "")],
            ["Otsu thresholds t₁ … t_K−1", ", ".join(f"{t:g}" for t in th) or "—"],
            ["pixels with I ≥ t_K−1", _n(si.get("_cand_px", {}).get("top_class", 0))],
            ["after opening (C)", _n(si.get("_cand_px", {}).get("opened", 0))]]
            + ([["removed by anatomical exclusion", _n(si["excluded_px"])]] if si.get("excluded_px") else [])
            + [["connected components of C", str(si.get("n_raw", 0))]])
    comps = si.get("components", [])
    b("sub", f"PROCESS · component filter (Eq. 3.2):  κᵢ = depth(xᵢ*) − ρᵢ,  keep if |Cᵢ| ≥ {cfg.min_cand_area} and κᵢ ≥ {cfg.min_clearance:g}")
    small = si.get("n_raw", 0) - len(comps)
    if comps:
        b.table(["id", "area", "ρᵢ", "depth(xᵢ*)", "κᵢ", "decision"],
                [[c["id"], c["area"], _f(c["radius"], 1), _f(c["radius"] + c["clearance"], 1), _f(c["clearance"], 1),
                  "rejected: on outline" if c.get("rejected") else "kept"] for c in comps],
                f"{small} further component(s) smaller than A_min = {cfg.min_cand_area} px were discarded first." if small else None)
    ranked = si.get("ranked", [])
    if ranked and ranked[0].get("_terms"):
        # The terms switched off contribute 0 (seed_ranking.score_candidates), so the
        # equation shows only the active ones, with the weights of this run.
        wc = cfg.rank_w_contrast if cfg.rank_use_contrast else 0.0
        ws = cfg.rank_w_shape if cfg.rank_use_shape else 0.0
        terms = ([f"{wc:g}·Δᵢ′"] if cfg.rank_use_contrast else []) + [f"{cfg.rank_w_radius:g}·ρᵢ′"] \
            + ([f"{ws:g}·sᵢ"] if cfg.rank_use_shape else [])
        off = [n for n, on in (("contrast", cfg.rank_use_contrast), ("shape", cfg.rank_use_shape)) if not on]
        b("sub", "PROCESS · ranking (Eq. 3.3)")
        b("eq", ["Qᵢ = w_c·Δᵢ′ + w_r·ρᵢ′ + w_s·sᵢ",
                 "Qᵢ = " + " + ".join(terms) + ("   (" + ", ".join(off) + " term off)" if off else ""),
                 "Δ′, ρ′ rescaled to [0, 1] across the kept candidates; s = (compactness + solidity)/2"]
                + ([f"Qᵢ −= 10 if the vesselness fraction of Cᵢ > {cfg.rank_vessel_max:g} (veto)"]
                   if cfg.rank_use_vesselness else []))
        rows = []
        for r in ranked:
            t = r["_terms"]
            check = wc * t["contrast_n"] + cfg.rank_w_radius * t["radius_n"] + ws * t["shape"] \
                - (10.0 if r.get("vetoed") else 0.0)
            rows.append([r["id"], _f(t["contrast"], 1), _f(t["contrast_n"], 3), _f(t["radius"], 1),
                         _f(t["radius_n"], 3), _f(t["shape"], 3), _f(check, 3), _f(t["score"], 3)]
                        + ([_f(r.get("vfrac"), 2)] if cfg.rank_use_vesselness else []))
        b.table(["id", "Δᵢ", "Δᵢ′", "ρᵢ", "ρᵢ′", "sᵢ", "Q from eq.", "Qᵢ (run)"]
                + (["vessel"] if cfg.rank_use_vesselness else []), rows, "Highest Qᵢ first.")
    elif ranked:
        b("sub", "PROCESS · ranking")
        b("eq", ["all ranking features off:  C* = largest inscribed radius ρᵢ (ties: brighter mean)"])
    ch = si.get("chosen")
    if ch:
        b("good", f"C* = component {ch['id']} (area {ch['area']} px)")
    elif si.get("status") != "OK":
        b("bad", f"{si.get('status')}: " + "; ".join(si.get("warnings", [])))
    b("sub", "OUTPUT")
    b("output", [("C", _arr(st.image, "mask"), "display: top-class pixels of H"),
                 ("C*", f"component {ch['id']}" if ch else "none", "5 · Seed core")])
    tips = {
        "c, η, K": "c = minimum clearance, η = interior fraction, K = number of Otsu classes. Tuned on the "
                   "BRISC 2025 training split and frozen (Table 3.4); K can be changed under Adjust Parameters.",
        "depth_max": "The largest depth(x) in H (Stage 2 distance map): how far the most central head pixel "
                     "lies from the head outline.",
        "depth cut": "Pixels shallower than this are scalp, skull or rim and are left out of the histogram; "
                     "the cut is at least c px and grows with the head size through η.",
        "interior pixels": "Pixels of H with depth(x) ≥ depth cut, read from the Stage 2 depth map. Only "
                           "these feed the Otsu histogram, because scalp fat outshines tumor on T1.",
        "κᵢ": "Clearance: depth of the component's deepest point minus its inscribed radius ρᵢ. Near 0 for "
              "scalp hugging the outline, large for an interior tumor.",
        "ρᵢ": "Inscribed radius: distance from the component's deepest point to its own edge.",
        "Δᵢ": "Contrast: mean of Cᵢ minus the mean of a ring around it; ′ marks the value rescaled to [0, 1].",
        "Q from eq.": "Qᵢ worked out by substituting the row into the equation above; it must equal 'Qᵢ (run)', "
                      "the score the pipeline ranked by.",
        "A_min": "Smallest component area accepted as a tumor candidate.",
    }
    if cfg.rank_use_vesselness:
        tips["vesselness"] = ("Frangi filter score of how tube-like each pixel is. A candidate whose pixels are "
                              "mostly tubular (enhancing vessels or sinuses) has 10 subtracted from Qᵢ.")
    if si.get("excluded_px"):
        tips["anatomical exclusion"] = ("Orbit exclusion removes bright, round components in the upper 40% of "
                                        "the head (eyeballs); skull-base exclusion removes shallow bright pixels "
                                        "in the bottom 12% of the head (marrow).")
    b("tips", tips)


# ─── Stage 5 · Seed core ──────────────────────────────────────────────────────
def _seed(result, cfg, b):
    st = result.stage("seed")
    si = st.info.get("_s_info", {})
    core = st.image
    co = si.get("_core")
    mode = result.meta.get("seed_mode")
    method = result.meta.get("method")
    b("sub", "INPUT")
    if mode == "auto":
        b("input", [("C*", f"{_n(co['comp_area'])} px" if co else "—", "4 · Candidates")])
    elif mode == "planted":
        b("input", [("planted core", f"{_n(co['comp_area'])} px" if co else f"{_n(core.sum())} px",
                     "evaluation protocol")])
    else:
        disk = si.get("_click_disk")
        b("input", [("click disk", f"{_n(disk.sum())} px" if disk is not None else "—",
                     _stage_name(result, "candidates", "operator clicks"))])
    if co:
        title = "PROCESS · purify manual seed" if co.get("purified") else "PROCESS · seed core (Section 3.2.1, step 2d)"
        b("sub", title)
        b("eq", ["dt(x) = distance from x to the boundary of the component",
                 f"S = {{ x : dt(x) ≥ α · dt_max }},  α = {cfg.seed_core_frac:g}"])
        b.table(["quantity", "value"], [
            ["dt_max", _f(co["dt_max"], 3)], ["α · dt_max", _f(co["cut"], 3)],
            ["component area", f"{_n(co['comp_area'])} px"], ["seed core |S|", f"{_n(core.sum())} px"]]
            + ([["fallback", "single deepest pixel"]] if co.get("fallback") else []))
    else:
        b("sub", "PROCESS · seed core")
        b("eq", ["S = click disk (used as clicked)"])
    if not core.any():
        b("bad", f"{st.info.get('status')}: no seed, so growth does not run.")
        return
    G, sym, _ = _growth_image(result)
    I = _working_image(result)
    px = _seed_pixels(result)
    b("sub", "COMPUTATION · seed-core pixels S")
    cols = ["row", "col", "I(x)"] + (["L(x)"] if sym == "L" else [])
    b.table(cols, [[r, c, _f(float(I[r, c]), 3)] + ([_f(float(G[r, c]), 6)] if sym == "L" else [])
                   for r, c in px[:SEED_CAP]],
            _seed_note(min(len(px), SEED_CAP), len(px)))
    vals = G[core]
    n, mu = int(vals.size), float(vals.mean())
    s = float(vals.std(ddof=1)) if n > 1 else 0.0
    b.table(["initial statistic", "value"], [
        ["n = |S|", _n(n)], [f"μ_A (mean of {sym} over S)", _f(mu, 6)], [f"s_A (SD of {sym} over S)", _f(s, 6)]])
    b("sub", "OUTPUT")
    to = "6 · Region growing"
    b("output", [("S", _arr(core, "mask"), f"{to} (initial region A = S)"),
                 ("n", _n(n), to), ("μ_A", _f(mu, 6), to), ("s_A", _f(s, 6), to)])
    tips = {
        "n = |S|": "The number of pixels in the seed core S.",
        "μ_A": TIP_MU_A,
        "s_A": TIP_S_A,
        "dt(x)": "Euclidean distance from x to the nearest pixel outside the component (its boundary).",
        "α": "Core fraction: only pixels at least α of the way from the boundary to the deepest point are "
             "kept, so partial-volume boundary pixels never enter the seed statistics.",
    }
    if co and co.get("purified"):
        tips["purify manual seed"] = (
            "ESRG with manual seeding only. Each click is grown into a disk of radius "
            f"{cfg.manual_seed_radius} px, which can straddle the tumor edge. Purification applies the "
            "automatic core's rule to it (keep dt ≥ α·dt_max), because boundary pixels inflate s_A, and a "
            "larger s_A loosens T_L = k_L·max(s_A, σ_floor) and lets the region leak.")
    b("tips", tips)


# ─── Stage 6 · Growth ─────────────────────────────────────────────────────────
GROWTH_TIPS = {
    "μ_loc": "Local reference: mean of L over the pixels already in the region inside the window W_r(y) "
             "around y. If none are there yet, μ_A is used.",
    "δ(y)": "Local difference |L(y) − μ_loc(y)|. It orders the sorted seed list (smallest first) and is "
            "compared with T_L.",
    "|L−μ_A|": "Distance of y from the region mean μ_A of this pass; compared with the drift guard T_G.",
    "T_L": "Local bound k_L·σ_A. The first popped pixel with δ > T_L ends the pass: the list is sorted, so "
           "every remaining pixel fails as well.",
    "T_G": "Global drift guard k_G·σ_A. A pixel within T_L but farther than T_G from μ_A is rejected for "
           "good; this stops slow chaining across a soft boundary.",
    "σ_A": "max(s_A, σ_floor), frozen for the whole pass so one borderline pixel cannot widen the bounds "
           "for the next.",
    "μ_A": TIP_MU_A,
    "s_A": TIP_S_A,
    "σ_floor": TIP_SIGMA_FLOOR,
    "P_max": "Maximum number of passes. Each pass re-estimates μ_A and σ_A from the larger region, which "
             "can widen the bounds; P_max caps how often. Most slices stop earlier, when a pass adds 0 pixels.",
    "Re-queued": "A pixel whose δ grew since it was queued (its local mean moved) is put back with the new "
                 "δ instead of being judged on the stale one.",
}


def _growth_esrg(result, cfg, b, st, tr):
    log = result.stage("log")
    b("sub", "INPUT")
    b("input", [("L", _arr(log.image), log.name),
                ("σ_floor", _f(log.info["_noise_floor"]["sigma_floor"], 6), log.name),
                ("S, n, μ_A, s_A", f"{_n(result.stage('seed').image.sum())} px", "5 · Seed core"),
                ("H", "growth never leaves H", "2 · Head mask")])
    if not cfg.use_log:
        b("note", "Log transform OFF for this run: L(x) = I(x), so every quantity below is in intensity units.")
    pd = tr.get("pass_detail", [])
    k_l = pd[0]["k_L"] if pd else cfg.k_local
    b("anchor", "obj2")
    b("sub", "PROCESS · per pixel y popped from the SSL (Eq. 3.8, 3.11)")
    eq = [f"μ_loc(y) = mean of L over A ∩ W_r(y),  r = {cfg.local_radius} ({2 * cfg.local_radius + 1}×{2 * cfg.local_radius + 1})"
          if cfg.use_log_local else "μ_loc(y) := μ_A   (ablation: global reference)",
          "δ(y) = | L(y) − μ_loc(y) |"]
    if cfg.use_stopping:
        eq += ["absorb y  ⇔  δ(y) ≤ T_L  ∧  |L(y) − μ_A| ≤ T_G",
               "δ(y) > T_L ends the pass (SSL is sorted);  |L(y) − μ_A| > T_G rejects y"]
    else:
        eq += ["absorb y unconditionally   (ablation: no stopping criterion)"]
    b("eq", eq)
    b("anchor", "obj3")
    b("sub", "PROCESS · per pass, statistics frozen (Eq. 3.9, 3.10)")
    b("eq", ["σ_A = max( s_A, σ_floor )",
             f"T_L = k_L · σ_A,   T_G = k_G · σ_A,   k_L = {k_l:g}, k_G = {cfg.k_global:g}"])
    if cfg.adaptive_k and "adaptive_k_local" in tr:
        b("note", f"Per-image adaptive k_L is ON: k_L = {tr['adaptive_k_local']:g} was set from the seed's "
                  f"contrast with its {cfg.adaptive_ring}-px ring (range {cfg.adaptive_k_lo:g}–{cfg.adaptive_k_hi:g}).")
    if pd:
        b("sub", "COMPUTATION · pass table")
        b.table(["p", "|A|", "μ_A", "s_A", "σ_A", "T_L", "T_G", "added", "rej."],
                [[d["pass"], _n(d["n"]), _f(d["mu"], 4), _f(d["s"], 4), _f(d["sigma"], 4), _f(d["T_L"], 4),
                  _f(d["T_G"], 4), _n(d["added"]), _n(d["rejected"])] for d in pd],
                f"P_max = {cfg.max_passes}. Re-queued (δ grew since queueing): "
                + ", ".join(f"p{d['pass']} {d['requeued']}" for d in pd) + ".")
    ev = tr.get("events")
    if ev is not None:
        for d in pd:
            pe = [e for e in ev if e[0] == d["pass"]]
            if not pe:
                continue
            shown = pe[:PIXEL_CAP]
            stop = next((e for e in pe if e[7] == "stop"), None)
            if stop is not None and stop not in shown:
                shown = shown + [stop]
            b("sub", f"COMPUTATION · pass {d['pass']} pixels (T_L {_f(d['T_L'], 4)}, T_G {_f(d['T_G'], 4)})")
            rows = []
            for e in shown:
                verdict = {"absorb": "absorb", "reject": "reject >T_G", "stop": "STOP >T_L"}[e[7]]
                rows.append([e[1], e[2], _f(e[3], 4), _f(e[4], 4), _f(e[5], 4), _f(e[6], 4), verdict])
            n_abs = sum(1 for e in pe if e[7] == "absorb")
            n_rej = sum(1 for e in pe if e[7] == "reject")
            b.table(["row", "col", "L(y)", "μ_loc", "δ(y)", "|L−μ_A|", "decision"], rows,
                    f"Pass {d['pass']}: {_n(n_abs)} absorbed, {_n(n_rej)} rejected"
                    + (f"; first {PIXEL_CAP} in SSL order shown" + (" plus the stopping pixel" if stop is not None and stop not in pe[:PIXEL_CAP] else "")
                       if len(pe) > PIXEL_CAP else "") + ".")
    b("kv", [("stop reason", str(tr.get("stop_reason")))])


def _growth_srg(result, cfg, b, st, tr):
    auto = result.meta.get("seed_mode") in ("auto", "planted")
    g, sym, src = _growth_image(result)
    local, stop = cfg.srg_use_log_local, cfg.srg_use_stopping
    b("sub", "INPUT")
    rows = [(f"g = {sym}" if sym == "L" else ("g = I·H" if auto else "g = I"), _arr(g), src)]
    if stop:
        rows.append(("σ_floor", _f(result.stage("log").info["_noise_floor"]["sigma_floor"], 6),
                     result.stage("log").name))
    rows.append(("S, n, μ_A, s_A", f"{_n(result.stage('seed').image.sum())} px", "5 · Seed core (region 1)"))
    if auto:
        rows.append(("background seeds", f"{tr.get('n_background_seeds')} on an {cfg.bg_seed_step}-px grid",
                     "inside H, ≥ 6 px from S"))
    else:
        rows.append(("seed regions", str(tr.get("seed_areas", "")), "operator clicks, types 1…n"))
    b("input", rows)
    applied = [n for n, on in (("log-domain local measure", local), ("adaptive termination", stop)) if on]
    b("anchor", "obj2")
    b("sub", "PROCESS · original SRG (Eq. 3.7), 4-connected"
             + (" + " + ", ".join(applied) + " on region 1" if applied else ", unconditional"))
    eq = ["δ(x) = | g(x) − mean_{y ∈ A} g(y) |   (A = adjacent region)"]
    if local:
        eq += [f"region 1:  δ(x) = | g(x) − μ_loc(x) |,  μ_loc = mean of g over A₁ ∩ W_r(x),  r = {cfg.local_radius}"]
    eq += ["pop the smallest δ from the SSL and give x to that region"]
    if stop:
        eq += ["region 1 takes x only if  δ(x) ≤ T_L  ∧  |g(x) − μ_A| ≤ T_G,  else x is left to the others",
               f"σ_A = max( s_A, σ_floor ),  T_L = k_L·σ_A,  T_G = k_G·σ_A,  k_L = {cfg.k_local:g}, k_G = {cfg.k_global:g}",
               "(SRG has no passes: μ_A and s_A are the running values at each decision)"]
    else:
        eq += ["no stopping rule"]
    eq += ["repeat until every pixel is allocated to some region"]
    b("eq", eq)
    b("anchor", "obj3")
    ev = tr.get("events")
    if ev:
        shown = ev[:PIXEL_CAP]
        b("sub", "COMPUTATION · region-1 decisions" if stop else "COMPUTATION · pixels absorbed into the tumor region (region 1)")
        ref = "μ_loc" if local else "mean_A"
        if stop:
            b.table(["row", "col", f"{sym}(x)", ref, "δ(x)", f"|{sym}−μ_A|", "T_L", "T_G", "decision"],
                    [[e[0], e[1], _f(e[2], 3), _f(e[3], 3), _f(e[4], 3), _f(e[5], 3), _f(e[6], 3), _f(e[7], 3), e[8]]
                     for e in shown],
                    None if len(shown) >= len(ev) else f"First {len(shown)} of {_n(len(ev))} region-1 decisions.")
        else:
            b.table(["row", "col", f"{sym}(x)", ref, "δ(x)"],
                    [[e[0], e[1], _f(e[2], 3), _f(e[3], 3), _f(e[4], 3)] for e in shown],
                    None if len(shown) >= len(ev) else f"First {len(shown)} of {_n(len(ev))} region-1 absorptions.")
    rows = [["region-1 area", f"{_n(st.image.sum())} px"]]
    if tr.get("declined") is not None and stop:
        rows.append(["pixels declined by region 1", _n(tr["declined"])])
    if tr.get("allocated") is not None:
        rows.append(["pixels allocated / in H", f"{_n(tr['allocated'])} / {_n(tr['mask_px'])}"])
    if tr.get("region_areas"):
        rows.append(["region areas", str(tr["region_areas"])])
        rows.append(["region means", str(tr["region_means"])])
    b.table(["quantity", "value"], rows)
    b("kv", [("stop reason", str(tr.get("stop_reason")))])
    if tr.get("warning"):
        b("bad", tr["warning"])
    tips = {"mean_A": "Mean of g over the tumor region at the moment x was scored.",
            "δ(x)": "Difference between x and the region it is compared with; the SSL always pops the "
                    "smallest δ of all regions next.",
            "background seeds": "With automatic seeding SRG has no operator-planted competitors, so a grid of "
                                "single-pixel seeds inside H plays that role."}
    if local:
        tips["μ_loc"] = GROWTH_TIPS["μ_loc"].replace("L over", "g over")
    if stop:
        tips.update({"T_L": "Local bound k_L·σ_A from the tumor region's running statistics.",
                     "T_G": "Drift guard k_G·σ_A around the running region mean μ_A.",
                     "declined": "Region 1 refused the pixel; it stays available to the background regions "
                                 "and is never offered to region 1 again.",
                     "σ_A": "max(s_A, σ_floor) of the tumor region, read at each decision.",
                     "σ_floor": TIP_SIGMA_FLOOR})
    b("tips", tips)


def _growth(result, cfg, b):
    st = result.stage("growth")
    tr = st.info.get("_trace", {})
    if result.meta.get("method") == "esrg":
        _growth_esrg(result, cfg, b, st, tr)
        b("tips", GROWTH_TIPS)
    else:
        _growth_srg(result, cfg, b, st, tr)
    if tr.get("events") is None:
        b("note", "Pixel-level log not recorded for this run.")
    b("sub", "OUTPUT")
    b("output", [("A", _arr(st.image, "mask"), "7 · Post-processing")])


def _tessellation(result, cfg, b):
    st = result.stage("tessellation")
    b("sub", "PROCESS · full SRG tessellation (Eq. 3.7)")
    b("note", "Every planted region grows under the same rule until no pixel is unallocated; "
              "region 1 (the tumor click) is passed on to Stage 6.")
    b.table(["region", "area (px)", "mean g"],
            [[k, _n(v), _f(st.info["region means"][k], 2)] for k, v in st.info["region areas"].items()])
    b("sub", "OUTPUT")
    b("output", [("labels", f"{st.info['seed regions']} regions", "6 · region 1 = A")])


# ─── Stage 7 · Final ──────────────────────────────────────────────────────────
def _final(result, cfg, b):
    st = result.stage("final")
    i = st.info
    if "before" not in i:
        b("bad", f"{result.status}: no seed, so no region was grown.")
        b("output", [("M", "empty mask", "Evaluation")])
        return
    b("sub", "INPUT")
    b("input", [("A", f"{_n(i['before'])} px", "6 · Region growing"), ("S", "seed core", "5 · Seed core")])
    b("sub", "PROCESS · post-processing (Section 3.2.1, step 4)")
    b("eq", ["A₁ = fill_holes(A)",
             f"A₂ = open(A₁, disk {cfg.post_open_radius})   (skipped if it would empty A₁)",
             "M  = connected component(s) of A₂ that intersect S"])
    b("sub", "COMPUTATION · whole region")
    b.table(["step", "area (px)", "change"], [
        ["A (grown)", _n(i["before"]), ""],
        ["A₁ holes filled", _n(i["after_fill"]), f"{i['after_fill'] - i['before']:+,}"],
        ["A₂ opened", _n(i["after_open"]), f"{i['after_open'] - i['after_fill']:+,}"],
        ["M = components touching S", _n(i["after"]), f"{i['after'] - i['after_open']:+,}"]],
        f"{i.get('components_dropped', 0)} component(s) not touching S were dropped.")
    hp, frac = i.get("_head_px"), i.get("_brain_frac")
    if hp:
        b("eq", [f"status = WARN  if |M| / |H| > {cfg.leak_warn_frac:.2f}"])
        line = f"|M| / |H| = {_n(i['after'])} / {_n(hp)} = {frac:.3f}  →  {st.status}"
        b("good" if st.status == "OK" else "bad", line)
    b("sub", "OUTPUT")
    b("output", [("M", _arr(st.image, "mask"), "Evaluation (compared with the ground truth G)")])
    tips = {
        "fill_holes": "Every background pixel that cannot reach the image border through background becomes "
                      "part of the region. This fills enclosed holes such as the necrotic core of a "
                      "ring-enhancing tumor.",
        "open(": f"Opening = erosion then dilation with a disk of radius {cfg.post_open_radius}. It cuts off "
                 "spurs and bridges too thin to survive the erosion and keeps the rest of the shape.",
        "components touching S": "After the opening, A₂ is split into connected components; only those "
                                 "containing at least one seed-core pixel are kept as the final mask M.",
        "WARN": "Diagnostic only, it does not change M: a mask covering this much of the head is a probable leak.",
    }
    if _srg_manual(result):
        tips["|H|"] = ("H is the head mask. SRG with manual seeding does not use it to grow; it is computed only "
                       "for this leakage warning.")
    b("tips", tips)


# ─── Evaluation ───────────────────────────────────────────────────────────────
def _evaluation(result, cfg, b, chosen):
    meta = result.meta
    gt = meta.get("ground_truth")
    c = evalx.counts(result)
    b("sub", "INPUT")
    b("input", [("M", f"{_n(result.mask.sum())} px", "7 · Final"),
                ("G", f"{_n(c['G'])} px  ({_name(gt)})" if c else "none found", "ground-truth mask (BRISC 2025)"),
                ("S", f"{_n(result.stage('seed').image.sum())} px", "5 · Seed core (seed hit)")])
    b("note", f"{str(meta.get('method', '')).upper()} · {meta.get('seed_mode')} seeding · status {result.status}. "
              "Every value is recomputed from the masks of this run (Eq. 3.32–3.39).")
    if not chosen:
        b("note", "No metric selected — tick one above.")
        return
    if c:
        b("sub", "PIXEL COUNTS · M compared with G pixel by pixel")
        b("text", "\n".join(evalx.count_lines(c)))
    entries = evalx.explain(result, cfg.leak_ratio)
    for k in chosen:
        e = entries[k]
        v = e["value"]
        defined = v is not None and not (isinstance(v, float) and np.isnan(v))
        ok = defined and not any(w in e["verdict"] for w in ("MISS", "LEAKED", "below", "undefined", "requires"))
        b("head", f"{e['label']}  {evalx.value_text(e)}")
        b("note", e["objective"])
        b("good" if ok else "bad", "→ " + e["verdict"])
        if e["formula"]:
            b("eq", e["formula"])
        b("sub", "COMPUTATION")
        b("text", "\n".join(e["steps"]))
        b("output", [(e["label"], evalx.value_text(e), "result")])
    b("tips", {"TP": "True positives: pixels in both M and G.",
               "FP": "False positives: pixels in M but not in G (spill outside the tumor).",
               "FN": "False negatives: pixels in G but not in M (tumor missed).",
               "TN": "True negatives: pixels in neither; no metric here uses them."})


STAGES = {"input": _input, "mask": _mask, "n4": _n4, "log": _log, "candidates": _candidates,
          "seed": _seed, "tessellation": _tessellation, "growth": _growth, "final": _final}


def report(result, cfg, key, metrics=None):
    """Display blocks for stage tab `key` of `result` (see module docstring).

    The configuration shown and used is the one the run was executed with
    (result.meta["_cfg"]), so toggled methods and adjusted hyperparameters are
    reflected exactly even if the controls have changed since; `cfg` is only a
    fallback for results produced without it."""
    cfg = result.meta.get("_cfg") or cfg
    b = _B()
    if key == "evaluation":
        b("head", "Evaluation")
        _evaluation(result, cfg, b, metrics or [])
        return b.out
    st = result.stage(key)
    if st is None or key not in STAGES:
        b("note", "No stage details for this view.")
        return b.out
    b("head", st.name)
    b("note", f"status {st.status}")
    STAGES[key](result, cfg, b)
    return b.out
