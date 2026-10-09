"""
stage_report.py — End-to-end, stage-by-stage computation report for the GUI.

Purpose : Let a reviewer verify, for the slice just run, that every stage does what
          Chapter 3 states: what enters the stage and from where, the equation and
          parameters applied, the computed values, and what leaves for which stage.
Function : report(result, cfg, key, metrics) returns a list of display blocks
          (kind, payload) for one stage tab. Repeated per-pixel work is shown as the
          equation once, then a table of inputs and results; whole-image work shows
          the image-level quantities. Values are read from the arrays and traces the
          run itself produced (pipeline.run(record=True)), never re-derived by a
          second implementation, except where a one-line check is the point.
Notes   : Pure NumPy, no GUI. Block kinds: flow, head, sub, input, output, eq, kv,
          table, text, note, good, bad. Tables are kept within ~66 characters so they
          fit the widened Details panel without wrapping.
"""
import os

import numpy as np

from . import explain as evalx
from . import pixel_report
from .config import Config

DEFAULTS = Config()

# Ablation switches and hyperparameters each stage reads, in display order.
# (field, label); "esrg" marks switches that only the enhanced grower uses.
SETTINGS = {
    "input": [("max_side", "longer side resized to (px)")],
    "mask": [("frame_open_radius", "frame-removal opening radius (px)")],
    "n4": [("use_n4", "N4 bias correction")],
    "log": [("use_log", "log-domain transform (Obj 2)"), ("log_eps", "ε"),
            ("sigma_floor_min", "lower clamp of σ_floor")],
    "candidates": [("seed_mode", "seeding"), ("otsu_classes", "K, Otsu classes"),
                   ("interior_frac", "η, interior fraction"), ("min_clearance", "c, minimum clearance (px)"),
                   ("min_cand_area", "A_min (px)"), ("rank_use_contrast", "rank by contrast"),
                   ("rank_w_contrast", "w_c"), ("rank_w_radius", "w_r"), ("rank_use_shape", "rank by shape"),
                   ("rank_w_shape", "w_s"), ("rank_use_vesselness", "vesselness veto"),
                   ("exclude_orbits", "orbit exclusion"), ("exclude_skull_base", "skull-base exclusion"),
                   ("manual_seed_radius", "click-disk radius (px)")],
    "seed": [("seed_mode", "seeding"), ("seed_core_frac", "α, core fraction"),
             ("purify_manual_seed", "purify manual seed (ESRG only)")],
    "tessellation": [("method", "method")],
    "growth": [("method", "method"), ("use_log", "log-domain transform (Obj 2)"),
               ("use_log_local", "local log measure (Obj 2)"), ("use_stopping", "adaptive termination (Obj 3)"),
               ("k_local", "k_L"), ("k_global", "k_G"), ("local_radius", "r, window radius"),
               ("max_passes", "P_max"), ("adaptive_k", "per-image adaptive k_L"),
               ("bg_seed_step", "SRG background-seed grid (px)")],
    "final": [("post_open_radius", "opening radius (px)"), ("leak_warn_frac", "WARN when |M|/|H| >")],
    "evaluation": [("leak_ratio", "λ, leakage ratio")],
}
ESRG_ONLY = {"use_log", "use_log_local", "use_stopping", "k_local", "k_global", "local_radius",
             "max_passes", "adaptive_k", "purify_manual_seed"}
SRG_ONLY = {"bg_seed_step"}


def _val(v):
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    return f"{v:g}" if isinstance(v, float) else str(v)


def _settings(cfg, key, method):
    """(label, value, changed, note) for every setting of the stage, read from the run's config."""
    rows = []
    for field, label in SETTINGS.get(key, []):
        v, dv = getattr(cfg, field), getattr(DEFAULTS, field)
        note = ""
        if method == "srg" and field in ESRG_ONLY:
            note = "not used by SRG"
        elif method == "esrg" and field in SRG_ONLY:
            note = "not used by ESRG"
        rows.append((label, _val(v), v != dv, note or (f"default {_val(dv)}" if v != dv else "")))
    return rows

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
def _flow(result, key):
    steps = [(s.key, s.name) for s in result.stages]
    steps.append(("evaluation", "Eval"))
    return ("flow", [(k, n, k == key) for k, n in steps])


def _seed_pixels(result):
    seed = result.stage("seed")
    if seed is None or not seed.image.any():
        return []
    return [(int(r), int(c)) for r, c in np.argwhere(seed.image)]


def _cap_note(shown, total, what="pixels"):
    return None if shown >= total else f"Showing {shown} of {_n(total)} {what}; the rest follow the same equation."


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
        px = _seed_pixels(result)[:SEED_CAP]
        if px:
            b("sub", "COMPUTATION · seed pixels (same equation for every pixel)")
            rows = []
            for r, c in px:
                v = float(raw[r, c])
                calc = min(max((v - lo) / (hi - lo), 0.0), 1.0) * 255.0
                rows.append([r, c, _f(v, 1), _f(calc, 3), _f(float(I[r, c]), 3)])
            b.table(["row", "col", "I₀(x)", "computed I(x)", "pipeline I(x)"], rows,
                    _cap_note(len(px), len(_seed_pixels(result))))
    b("sub", "OUTPUT")
    b("output", [("I", _arr(I), f"2 · Head mask (and the image under every later stage)")])


# ─── Stage 2 · Head mask ──────────────────────────────────────────────────────
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
    rad, drop = st.info.get("radius"), st.info.get("area_drop")
    b("note", f"Granulometry (diagnostic only, mask unchanged): sharpest area loss at r = {rad}, "
              f"drop {drop:.1%} of H; skull ring {'found' if st.info.get('ring_found') else 'not found'}.")
    for w in st.info.get("warnings", []):
        b("bad", w)
    b("sub", "OUTPUT")
    b("output", [("H", _arr(H, "mask"), f"3 · Log (I·H), 4 · Candidates, 6 · Growth (never leaves H), 7 · Final"),
                 ("depth", f"max {_f(s.get('depth_max'), 1)} px", f"4 · Candidates (interior rule, Eq. 3.2)"),
                 ("I·H", "I with every x ∉ H set to 0", f"3 · Log domain")])


# ─── Stage 2b · N4 (ablation) ─────────────────────────────────────────────────
def _n4(result, cfg, b):
    st = result.stage("n4")
    b("sub", "PROCESS · N4ITK bias correction (ablation arm only, not part of ESRG)")
    b("eq", ["I_N4(x) = (I(x) + 1) / b̂(x) − 1,  then renormalized to [0, 255]"])
    b("kv", [(k, str(v)) for k, v in st.info.items() if not k.startswith("_")])
    b("sub", "OUTPUT")
    b("output", [("I_N4", _arr(st.image), f"3 · Log domain, replaces I·H")])


# ─── Stage 3 · Log domain + noise floor ───────────────────────────────────────
def _log(result, cfg, b):
    st = result.stage("log")
    mask = result.stage("mask").image
    src = result.stage("n4") or result.stage("input")
    I = np.where(mask, src.image, 0.0)
    L, use_log, eps, nf = st.image, st.info["_use_log"], st.info["_eps"], st.info["_noise_floor"]
    b("sub", "INPUT")
    b("input", [("I·H", _arr(I), "1 · Input × 2 · Head mask"), ("H", _arr(mask, "mask"), "2 · Head mask")])
    b("sub", "PROCESS · log transform (Eq. 3.6)" if use_log else "PROCESS · log transform OFF (ablation)")
    b("eq", [f"L(x) = ln( I(x) + ε ),  ε = {eps:g}" if use_log else "L(x) = I(x)   (pixels pass through)"])
    px = _seed_pixels(result)
    if px:
        g = pixel_report.log_domain([("seed", px[:SEED_CAP])], I, L, mask, eps, use_log)[0]
        b("sub", "COMPUTATION · seed pixels (same equation for every pixel of H)")
        b.table(["row", "col", "I(x)", "I(x)+ε", "L(x)"],
                [[p["row"], p["col"], _f(p["I"], 4), _f(p["I"] + (eps if use_log else 0), 4), _f(p["L"], 6)]
                 for p in g["pixels"]], _cap_note(len(g["pixels"]), len(px)))
    b("sub", "PROCESS · noise floor σ_floor (Section 3.2.1, step 1e)")
    b("eq", ["r(x) = L(x) − median₃ₓ₃(L)(x),   x ∈ H",
             "MAD = median | r − median(r) |",
             f"σ_floor = max( 1.4826 · MAD, {nf['min_value']:g} )"])
    b("sub", "COMPUTATION · whole head")
    b.table(["quantity", "value"], [
        ["pixels of H used", _n(nf["n"])], ["median(r)", _f(nf["median_resid"], 6)],
        ["MAD", _f(nf["mad"], 6)], ["1.4826 · MAD", _f(nf["raw_sigma"], 6)],
        ["σ_floor", _f(nf["sigma_floor"], 6)]])
    b("sub", "OUTPUT")
    b("output", [("L", _arr(L), f"6 · Region growing (δ, μ_A, s_A)"),
                 ("σ_floor", _f(nf["sigma_floor"], 6), f"6 · lower bound of σ_A (Eq. 3.9)")])


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
        b("output", [("click disk", _arr(st.image, "mask"), f"5 · Seed core")])
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
                + (["vessel"] if cfg.rank_use_vesselness else []), rows,
                "Highest Qᵢ first. “Q from eq.” substitutes the row into the equation above; "
                "“Qᵢ (run)” is the score the pipeline ranked by. The two columns must agree.")
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
                 ("C*", f"component {ch['id']}" if ch else "none", f"5 · Seed core")])


# ─── Stage 5 · Seed core ──────────────────────────────────────────────────────
def _seed(result, cfg, b):
    st = result.stage("seed")
    si = st.info.get("_s_info", {})
    core = st.image
    co = si.get("_core")
    mode = result.meta.get("seed_mode")
    b("sub", "INPUT")
    b("input", [("C*" if mode == "auto" else "click disk",
                 f"{_n(co['comp_area'])} px" if co else "—",
                 "4 · Candidates" if mode == "auto" else "operator clicks")])
    b("sub", "PROCESS · seed core (Section 3.2.1, step 2d)")
    if co:
        b("eq", ["dt(x) = distance from x to the boundary of the component",
                 f"S = {{ x : dt(x) ≥ α · dt_max }},  α = {cfg.seed_core_frac:g}"])
        b.table(["quantity", "value"], [
            ["dt_max", _f(co["dt_max"], 3)], ["α · dt_max", _f(co["cut"], 3)],
            ["component area", f"{_n(co['comp_area'])} px"], ["seed core |S|", f"{_n(core.sum())} px"]]
            + ([["fallback", "single deepest pixel"]] if co.get("fallback") else []))
    else:
        b("eq", ["S = click disk (no purification for this method)"])
    if not core.any():
        b("bad", f"{st.info.get('status')}: no seed, so growth does not run.")
        return
    tr = (result.stage("growth").info.get("_trace", {}) if result.stage("growth") else {})
    L = result.stage("log").image
    I = np.where(result.stage("mask").image, result.stage("input").image, 0.0)
    px = _seed_pixels(result)
    b("sub", "COMPUTATION · seed pixels")
    b.table(["row", "col", "I(x)", "L(x)"],
            [[r, c, _f(float(I[r, c]), 3), _f(float(L[r, c]), 6)] for r, c in px[:SEED_CAP]],
            _cap_note(min(len(px), SEED_CAP), len(px)))
    ss = tr.get("seed_stats")
    if ss:
        vals = L[core] if result.meta.get("method") == "esrg" else I[core]
        sym = "L" if result.meta.get("method") == "esrg" else "I"
        b.table(["initial statistic", "value"], [
            ["n = |S|", _n(ss["n"])], [f"mean of {sym} over S (μ_A)", _f(ss["mean"], 6)],
            [f"check: mean computed from table", _f(float(vals.mean()), 6)]]
            + ([["s_A (sample SD)", _f(ss["s"], 6)]] if "s" in ss else []))
    b("sub", "OUTPUT")
    b("output", [("S", _arr(core, "mask"), f"6 · initial region A = S")])


# ─── Stage 6 · Growth ─────────────────────────────────────────────────────────
def _growth_esrg(result, cfg, b, st, tr):
    b("sub", "INPUT")
    b("input", [("L", _arr(result.stage("log").image), "3 · Log domain"),
                ("σ_floor", _f(result.stage("log").info["_noise_floor"]["sigma_floor"], 6), "3 · Log domain"),
                ("S", f"{_n(result.stage('seed').image.sum())} px", "5 · Seed core"),
                ("H", "growth never leaves H", "2 · Head mask")])
    if not cfg.use_log:
        b("note", "Log transform OFF for this run: L(x) = I(x), so every quantity below is in intensity units.")
    pd = tr.get("pass_detail", [])
    k_l = pd[0]["k_L"] if pd else cfg.k_local
    b("sub", "PROCESS · per pass, statistics frozen (Eq. 3.9, 3.10)")
    b("eq", ["σ_A = max( s_A, σ_floor )",
             f"T_L = k_L · σ_A,   T_G = k_G · σ_A,   k_L = {k_l:g}, k_G = {cfg.k_global:g}"])
    if cfg.adaptive_k and "adaptive_k_local" in tr:
        b("note", f"Per-image adaptive k_L is ON: k_L = {tr['adaptive_k_local']:g} was set from the seed's "
                  f"contrast with its {cfg.adaptive_ring}-px ring (range {cfg.adaptive_k_lo:g}–{cfg.adaptive_k_hi:g}).")
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
    if pd:
        b("sub", "COMPUTATION · pass table")
        b.table(["p", "|A|", "μ_A", "s_A", "σ_A", "T_L", "T_G", "added", "rej."],
                [[d["pass"], _n(d["n"]), _f(d["mu"], 4), _f(d["s"], 4), _f(d["sigma"], 4), _f(d["T_L"], 4),
                  _f(d["T_G"], 4), _n(d["added"]), _n(d["rejected"])] for d in pd],
                f"Max passes P_max = {cfg.max_passes}. Re-queued (δ grew since queueing): "
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
    I = np.where(result.stage("mask").image, result.stage("input").image, 0.0)
    b("sub", "INPUT")
    b("input", [("g = I·H" if auto else "g = I", _arr(I if auto else result.stage("input").image), "1 · Input (SRG has no log domain)"),
                ("S", f"{_n(result.stage('seed').image.sum())} px", "5 · Seed core (region 1)")]
      + ([("background seeds", f"{tr.get('n_background_seeds')} on an {cfg.bg_seed_step}-px grid", "inside H, ≥ 6 px from S")]
         if auto else [("seed regions", str(tr.get("seed_areas", "")), "operator clicks, types 1…n")]))
    b("sub", "PROCESS · original SRG (Eq. 3.7), 4-connected, unconditional")
    b("eq", ["δ(x) = | g(x) − mean_{y ∈ A} g(y) |   (A = adjacent region)",
             "pop the smallest δ from the SSL and absorb it — no stopping rule",
             "repeat until every pixel is allocated to some region"])
    ev = tr.get("events")
    if ev:
        shown = ev[:PIXEL_CAP]
        b("sub", "COMPUTATION · pixels absorbed into the tumor region (region 1)")
        b.table(["row", "col", "g(x)", "mean_A", "δ(x)"],
                [[e[0], e[1], _f(e[2], 3), _f(e[3], 3), _f(e[4], 3)] for e in shown],
                _cap_note(len(shown), len(ev), "region-1 absorptions"))
    rows = [["region-1 area", f"{_n(st.image.sum())} px"]]
    if tr.get("allocated") is not None:
        rows.append(["pixels allocated / in H", f"{_n(tr['allocated'])} / {_n(tr['mask_px'])}"])
    if tr.get("region_areas"):
        rows.append(["region areas", str(tr["region_areas"])])
        rows.append(["region means", str(tr["region_means"])])
    b.table(["quantity", "value"], rows)
    b("kv", [("stop reason", str(tr.get("stop_reason")))])
    if tr.get("warning"):
        b("bad", tr["warning"])


def _growth(result, cfg, b):
    st = result.stage("growth")
    tr = st.info.get("_trace", {})
    if result.meta.get("method") == "esrg":
        _growth_esrg(result, cfg, b, st, tr)
    else:
        _growth_srg(result, cfg, b, st, tr)
    if tr.get("events") is None:
        b("note", "Pixel-level log not recorded for this run.")
    b("sub", "OUTPUT")
    b("output", [("A", _arr(st.image, "mask"), f"7 · Post-processing")])


def _tessellation(result, cfg, b):
    st = result.stage("tessellation")
    b("sub", "PROCESS · full SRG tessellation (Eq. 3.7)")
    b("note", "Every planted region grows under the same rule until no pixel is unallocated; "
              "region 1 (the tumor click) is passed on to Stage 6.")
    b.table(["region", "area (px)", "mean g"],
            [[k, _n(v), _f(st.info["region means"][k], 2)] for k, v in st.info["region areas"].items()])
    b("sub", "OUTPUT")
    b("output", [("labels", f"{st.info['seed regions']} regions", f"6 · region 1 = A")])


# ─── Stage 7 · Final ──────────────────────────────────────────────────────────
def _final(result, cfg, b):
    st = result.stage("final")
    i = st.info
    if "before" not in i:
        b("bad", f"{result.status}: no seed, so no region was grown.")
        b("output", [("M", "empty mask", f"Evaluation")])
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
        ["M kept with S", _n(i["after"]), f"{i['after'] - i['after_open']:+,}"]],
        f"{i.get('components_dropped', 0)} component(s) not touching S were dropped.")
    hp, frac = i.get("_head_px"), i.get("_brain_frac")
    if hp:
        b("eq", [f"status = WARN  if |M| / |H| > {cfg.leak_warn_frac:.2f}"])
        line = f"|M| / |H| = {_n(i['after'])} / {_n(hp)} = {frac:.3f}  →  {st.status}"
        b("good" if st.status == "OK" else "bad", line)
    b("sub", "OUTPUT")
    b("output", [("M", _arr(st.image, "mask"), f"Evaluation (compared with the ground truth G)")])


# ─── Evaluation ───────────────────────────────────────────────────────────────
def _evaluation(result, cfg, b, chosen):
    meta = result.meta
    gt = meta.get("ground_truth")
    b("sub", "INPUT")
    b("input", [("M", f"{_n(result.mask.sum())} px", "7 · Final"),
                ("G", _name(gt) if gt else "none found", "ground-truth mask (BRISC 2025)"),
                ("S", f"{_n(result.stage('seed').image.sum())} px", "5 · Seed core (seed hit)")])
    b("note", f"{str(meta.get('method', '')).upper()} · {meta.get('seed_mode')} seeding · status {result.status}. "
              "Every value is recomputed from the masks of this run (Eq. 3.32–3.39).")
    if not chosen:
        b("note", "No metric selected — tick one above.")
        return
    entries = evalx.explain(result, cfg.leak_ratio)
    for k in chosen:
        e = entries[k]
        v = e["value"]
        defined = v is not None and not (isinstance(v, float) and np.isnan(v))
        ok = defined and not any(w in e["verdict"] for w in ("MISS", "LEAKED", "below", "undefined", "requires"))
        b("head", f"{e['label']}  {evalx.value_text(e)}")
        b("note", e["objective"])
        b("good" if ok else "bad", "→ " + e["verdict"])
        if e["sources"]:
            b("sub", "SOURCES OF THE VARIABLES")
            b("kv", [(f"{sym}  {val}", origin) for sym, val, origin in e["sources"]])
        if e["formula"]:
            b("eq", e["formula"])
        b("sub", "COMPUTATION")
        b("text", "\n".join(e["steps"]))
        b("output", [(e["label"], evalx.value_text(e), "result")])


STAGES = {"input": _input, "mask": _mask, "n4": _n4, "log": _log, "candidates": _candidates,
          "seed": _seed, "tessellation": _tessellation, "growth": _growth, "final": _final}


def report(result, cfg, key, metrics=None):
    """Display blocks for stage tab `key` of `result` (see module docstring).

    The configuration shown and used is the one the run was executed with
    (result.meta["_cfg"]), so toggled ablations and adjusted hyperparameters are
    reflected exactly even if the controls have changed since; `cfg` is only a
    fallback for results produced without it."""
    cfg = result.meta.get("_cfg") or cfg
    method = result.meta.get("method", cfg.method)
    b = _B()
    b(*_flow(result, key))
    if key == "evaluation":
        b("head", "Evaluation")
        b("settings", _settings(cfg, key, method))
        _evaluation(result, cfg, b, metrics or [])
        return b.out
    if key == "compare":
        b("head", "Compare")
        b("note", "Visual comparison of M (7 · Final) with the ground truth G; the numbers are in Evaluation.")
        return b.out
    st = result.stage(key)
    if st is None or key not in STAGES:
        b("note", "No stage details for this view.")
        return b.out
    b("head", st.name)
    b("note", f"status {st.status}")
    if SETTINGS.get(key):
        b("settings", _settings(cfg, key, method))
    STAGES[key](result, cfg, b)
    return b.out
