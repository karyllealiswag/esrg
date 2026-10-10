"""
pipeline.py — Stage orchestration.

Purpose : Run the stages in order and retain every intermediate result, so the GUI
          can inspect any stage and a bad final mask traces to its cause.
Function : run() executes input -> head mask -> log -> candidates -> seed -> growth
          -> post-process, wrapping each in a timed Stage record with diagnostics;
          it hard-fails to "no tumor candidate" when no seed survives, and attaches
          scores when a ground-truth mask is available.
Notes   : Each Stage keeps its display array and a status (OK/WARN/FAIL). The stage
          the GUI reads is not necessarily where a fact should be written elsewhere.
          Only the stages a run uses are listed: the SRG baseline lists Stage 3 only
          when an ESRG module is applied to it (cfg.srg_use_*), and under manual
          seeding it lists neither the head mask nor the candidates.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import preprocessing as pre
from . import seeds as seedmod
from . import growing, postprocess, metrics
from .io_utils import load_image, load_mask, find_mask_for, parse_brisc_name


@dataclass
class Stage:
    key: str                 # short id, e.g. "seed"
    name: str                # GUI button label
    image: Any               # array to display
    kind: str                # "gray" | "mask" | "heat" | "overlay" | "labels"
    info: dict = field(default_factory=dict)
    seconds: float = 0.0
    status: str = "OK"       # OK | WARN | FAIL


@dataclass
class Result:
    stages: list
    mask: Optional[np.ndarray] = None
    gt: Optional[np.ndarray] = None
    scores: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    status: str = "OK"

    def stage(self, key):
        return next((s for s in self.stages if s.key == key), None)


def run(image_path, cfg, mask_path=None, manual_points=None, progress=None,
        planted_core=None, raw_hook=None, record=False):
    """
    Full pipeline on one slice. manual_points overrides seed selection when
    cfg.seed_mode == 'manual'. progress(str) is an optional GUI callback.

    Evaluation-only hooks (experiments/evaluate.py):
      planted_core : boolean seed map that replaces seed selection. ESRG purifies
                     it exactly as a manual click (cfg.purify_manual_seed); SRG
                     grows it against the automatic background-seed grid.
      raw_hook     : function applied to the loaded slice before normalization,
                     e.g. to multiply in a synthetic bias field.

    GUI hook:
      record       : keep the intermediates the stage telemetry explains (normalization
                     percentiles, the raw slice, every growth decision). Results are
                     identical with it on or off; it only costs memory and time.
    """
    cfg.validate()
    stages, t_total = [], time.perf_counter()

    def say(msg):
        if progress:
            progress(msg)

    def timed(fn):
        t = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t

    # ── Stage 1: input ───────────────────────────────────────────────────────
    say("Loading image…")
    (raw, scale), dt = timed(lambda: load_image(image_path, cfg.max_side))
    if raw_hook is not None:
        raw = raw_hook(raw)
    img, norm_info = pre.normalize_details(raw)
    meta = {"path": image_path, "shape": img.shape, "scale": round(scale, 3),
            "method": cfg.method,
            "_cfg": cfg,  # the exact settings of this run, read back by the GUI telemetry
            "seed_mode": "planted" if planted_core is not None else cfg.seed_mode,
            **parse_brisc_name(image_path)}
    stages.append(Stage("input", "1 · Input", img, "gray",
                        {"size": f"{img.shape[1]} × {img.shape[0]} px",
                         "scale factor": round(scale, 3), "file": image_path,
                         "_norm": norm_info, **({"_raw": raw} if record else {})}, dt))

    # ── Stage 2: head mask ───────────────────────────────────────────────────
    say("Masking head…")
    (mask, depth, ss_info), dt = timed(lambda: pre.head_mask(img, cfg))
    # SRG with manual seeding grows on the whole image (the 1994 algorithm has no
    # skull stripping), so the head mask is not one of its stages; it is still
    # computed for the |M|/|H| leakage warning at the end.
    srg = cfg.method == "srg"
    srg_manual = srg and cfg.seed_mode == "manual" and planted_core is None
    if not srg_manual:
        stages.append(Stage("mask", "2 · Head mask", mask, "mask", ss_info, dt,
                            "WARN" if ss_info["warnings"] else "OK"))
    img_masked = np.where(mask, img, 0.0)

    # ── Stage 2b: N4 (ablation only) ─────────────────────────────────────────
    if cfg.use_n4:
        say("N4 bias correction…")
        try:
            (corrected, bias), dt = timed(lambda: pre.n4_correct(img_masked, mask))
            img_masked = pre.normalize(corrected) * mask
            stages.append(Stage("n4", "2b · N4 correction", img_masked, "gray",
                                {"bias range": f"{bias.min():.2f}–{bias.max():.2f}"}, dt))
        except Exception as e:
            stages.append(Stage("n4", "2b · N4 correction", img_masked, "gray",
                                {"error": str(e)}, 0.0, "FAIL"))

    # ── Stage 3: log domain ──────────────────────────────────────────────────
    # ESRG always lists it (with the transform off it still yields the noise floor).
    # SRG grows on intensity, over the whole image under manual seeding, unless the
    # Objective 2 module is applied, and needs the noise floor only for the
    # Objective 3 module; it lists Stage 3 only when one of them is on.
    if srg:
        base = img if srg_manual else img_masked
        nf_region = np.ones_like(mask) if srg_manual else mask
        use_log, show = cfg.srg_use_log_local, cfg.srg_use_log_local or cfg.srg_use_stopping
    else:
        base, nf_region, use_log, show = img_masked, mask, cfg.use_log, True
    if use_log:
        say("Log transform…")
        L, dt = timed(lambda: pre.log_transform(base, cfg.log_eps))
    else:
        # Ablation: growth and the noise floor run on the normalized intensity itself.
        L, dt = base.copy(), 0.0
    nf = pre.noise_floor_details(L, nf_region, cfg.sigma_floor_min) if show else None
    sigma_floor = nf["sigma_floor"] if nf else None
    if show:
        name = ("3 · Log domain" if use_log else "3 · Noise floor" if srg else "3 · Log domain (off)")
        stages.append(Stage("log", name, L, "heat",
                            {"transform": f"L = ln(I + {cfg.log_eps:g})" if use_log
                                          else "off — L = I (no log transform)",
                             "range": f"{L[nf_region].min():.2f}–{L[nf_region].max():.2f}",
                             "noise floor σ": round(sigma_floor, 4),
                             "note": "multiplicative bias becomes additive" if use_log
                                     else "multiplicative bias stays multiplicative",
                             "_use_log": use_log, "_eps": cfg.log_eps, "_noise_floor": nf,
                             "_region": nf_region, "_base": base}, dt))

    # ── Stage 4–5: seed selection ────────────────────────────────────────────
    say("Selecting seed…")
    if planted_core is not None:
        core = planted_core.astype(bool).copy()
        s_info = {"status": "OK" if core.any() else "NO SEED", "core_area": int(core.sum()),
                  "components": [], "warnings": []}
        if cfg.method == "esrg" and cfg.purify_manual_seed:
            core = seedmod.purify_core(core, cfg, s_info)
            s_info["core_area"] = int(core.sum())
        seed_labels = core.astype(np.int32)
        cand, dt = core, 0.0
    elif cfg.seed_mode == "manual":
        (seed_labels, s_info), dt = timed(lambda: seedmod.manual_seed(manual_points or [], img.shape, cfg))
        core = seed_labels == 1
        if cfg.method == "esrg" and cfg.purify_manual_seed:
            # SRG's seed_labels (used verbatim by grow_srg) is left untouched --
            # this only reshapes the ESRG growth core.
            core = seedmod.purify_core(core, cfg, s_info)
            s_info["core_area"] = int(core.sum())
        s_info["_click_disk"] = seed_labels == 1
        cand = core
    else:
        (core, s_info), dt = timed(lambda: seedmod.select_seed(img_masked, mask, depth, cfg))
        try:
            from skimage import filters as _f
            th = s_info.get("thresholds")
            cand = mask & (img_masked >= th[-1]) if th else np.zeros_like(mask)
        except Exception:
            cand = np.zeros_like(mask)

    if not srg_manual:  # SRG uses the clicks as they are: no candidate step
        stages.append(Stage("candidates", "4 · Candidates", cand, "mask",
                            {"thresholds": s_info.get("thresholds"),
                             "components found": s_info.get("n_raw"),
                             "components kept": s_info.get("n_kept"),
                             "detail": s_info.get("components", [])[:8], "_s_info": s_info}, dt))
    stages.append(Stage("seed", "5 · Seed core", core, "mask",
                        {"status": s_info["status"], "chosen": s_info.get("chosen"),
                         "core area": s_info.get("core_area"),
                         **({"seed types": s_info["types"]} if s_info.get("types") else {}),
                         "warnings": s_info.get("warnings", []), "_s_info": s_info}, 0.0,
                        "FAIL" if s_info["status"] != "OK" else "OK"))

    result = Result(stages=stages, meta=meta)

    if not core.any():
        # Hard fail: report no detection rather than seed arbitrarily.
        result.status = "NO TUMOR CANDIDATE"
        result.mask = np.zeros_like(mask)
        stages.append(Stage("final", "7 · Final", result.mask, "overlay",
                            {"status": result.status}, 0.0, "FAIL"))
        meta["seconds"] = round(time.perf_counter() - t_total, 3)
        return _attach_scores(result, mask_path, image_path, cfg, core)

    # ── Stage 6: region growing ──────────────────────────────────────────────
    say("Growing region…")
    # Applied ESRG modules for the SRG baseline (region 1 only; both None = 1994 SRG).
    srg_local = cfg.local_radius if cfg.srg_use_log_local else None
    srg_stop = (dict(k_L=cfg.k_local, k_G=cfg.k_global, sigma_floor=sigma_floor)
                if cfg.srg_use_stopping else None)
    if srg and not srg_manual:
        # No planted competitors: the auto core competes with a grid of
        # background seeds inside the head mask.
        (region, trace), dt = timed(lambda: growing.grow_srg_auto(L, mask, core, cfg, record,
                                                                     srg_local, srg_stop))
        growth_info = {"stop reason": trace["stop_reason"], "passes": trace["passes"],
                       "background seeds": trace["n_background_seeds"]}
    elif srg:
        # Whole image, unmasked: the published algorithm has no skull-stripping step
        # and leaves no pixel unallocated. Every planted id grows under the same
        # rule; grow_srg has no notion of "tumor" at all.
        (label_map, trace), dt = timed(lambda: growing.grow_srg(L, seed_labels, record, srg_local, srg_stop))
        stages.append(Stage("tessellation", "6a · Full tessellation", label_map, "labels",
                            {"seed regions": trace["n_seed_regions"],
                             "region areas": trace["region_areas"],
                             "region means": trace["region_means"]}, 0.0))
        # Id 1 is a labeling convention decided outside grow_srg: whichever seed
        # group the user planted on the structure of interest. The algorithm
        # itself grew every id identically.
        region = label_map == 1
        growth_info = {"stop reason": trace["stop_reason"],
                       "seed regions": trace["n_seed_regions"],
                       "unlabeled px": trace["unlabeled"],
                       **({"warning": trace["warning"]} if "warning" in trace else {})}
    else:
        (region, trace), dt = timed(lambda: growing.grow_esrg(L, mask, core, sigma_floor, cfg, record))
        growth_info = {"stop reason": trace["stop_reason"], "passes": trace["passes"]}
    growth_info["_trace"] = trace
    stages.append(Stage("growth", "6 · Region growing", region, "mask", growth_info, dt,
                        "WARN" if "warning" in trace else "OK"))

    # ── Stage 7: post-processing + final ─────────────────────────────────────
    say("Post-processing…")
    (final, p_info), dt = timed(lambda: postprocess.postprocess(region, core, cfg))
    result.mask = final

    brain_frac = final.sum() / max(mask.sum(), 1)
    p_info["_head_px"] = int(mask.sum())
    p_info["_brain_frac"] = float(brain_frac)
    status = "OK"
    if brain_frac > cfg.leak_warn_frac:
        status = "WARN"
        p_info["warning"] = (f"Mask covers {brain_frac:.0%} of the head region — "
                             "probable leakage.")
    stages.append(Stage("final", "7 · Final", final, "overlay", p_info, dt, status))

    meta["seconds"] = round(time.perf_counter() - t_total, 3)
    result.status = status
    return _attach_scores(result, mask_path, image_path, cfg, core)


def _attach_scores(result, mask_path, image_path, cfg, core):
    """Score against ground truth when a mask is available."""
    mp = mask_path or find_mask_for(image_path)
    if not mp:
        result.meta["ground_truth"] = None
        return result
    gt = load_mask(mp, result.mask.shape)
    result.gt = gt
    result.scores = metrics.evaluate(result.mask, gt, cfg)
    result.scores["seed_hit"] = metrics.seed_hit(core, gt)
    result.scores["seed_distance"] = metrics.seed_distance(core, gt)
    result.scores["seconds"] = result.meta.get("seconds")
    result.meta["ground_truth"] = mp
    return result
