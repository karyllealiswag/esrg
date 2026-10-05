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
        planted_core=None, raw_hook=None):
    """
    Full pipeline on one slice. manual_points overrides seed selection when
    cfg.seed_mode == 'manual'. progress(str) is an optional GUI callback.

    Evaluation-only hooks (experiments/evaluate.py):
      planted_core : boolean seed map that replaces seed selection. ESRG purifies
                     it exactly as a manual click (cfg.purify_manual_seed); SRG
                     grows it against the automatic background-seed grid.
      raw_hook     : function applied to the loaded slice before normalization,
                     e.g. to multiply in a synthetic bias field.
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
    img = pre.normalize(raw)
    meta = {"path": image_path, "shape": img.shape, "scale": round(scale, 3),
            "method": cfg.method,
            "seed_mode": "planted" if planted_core is not None else cfg.seed_mode,
            **parse_brisc_name(image_path)}
    stages.append(Stage("input", "1 · Input", img, "gray",
                        {"size": f"{img.shape[1]} × {img.shape[0]} px",
                         "scale factor": round(scale, 3), "file": image_path}, dt))

    # ── Stage 2: head mask ───────────────────────────────────────────────────
    say("Masking head…")
    (mask, depth, ss_info), dt = timed(lambda: pre.head_mask(img, cfg))
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
    if cfg.use_log:
        say("Log transform…")
        L, dt = timed(lambda: pre.log_transform(img_masked, cfg.log_eps))
    else:
        # Ablation: growth and the noise floor run on the normalized intensity itself.
        L, dt = img_masked.copy(), 0.0
    nf = pre.noise_floor_details(L, mask, cfg.sigma_floor_min)
    sigma_floor = nf["sigma_floor"]
    stages.append(Stage("log", "3 · Log domain" if cfg.use_log else "3 · Log domain (off)",
                        L, "heat",
                        {"transform": f"L = ln(I + {cfg.log_eps:g})" if cfg.use_log
                                      else "off — L = I (no log transform)",
                         "range": f"{L[mask].min():.2f}–{L[mask].max():.2f}",
                         "noise floor σ": round(sigma_floor, 4),
                         "note": "multiplicative bias becomes additive" if cfg.use_log
                                 else "multiplicative bias stays multiplicative",
                         "_use_log": cfg.use_log, "_eps": cfg.log_eps,
                         "_noise_floor": nf}, dt))

    # ── Stage 4–5: seed selection ────────────────────────────────────────────
    say("Selecting seed…")
    if planted_core is not None:
        core = planted_core.astype(bool).copy()
        s_info = {"status": "OK" if core.any() else "NO SEED", "core_area": int(core.sum()),
                  "components": [], "warnings": []}
        if cfg.method == "esrg" and cfg.purify_manual_seed:
            core = seedmod.purify_core(core, cfg)
            s_info["core_area"] = int(core.sum())
        seed_labels = core.astype(np.int32)
        cand, dt = core, 0.0
    elif cfg.seed_mode == "manual":
        (seed_labels, s_info), dt = timed(lambda: seedmod.manual_seed(manual_points or [], img.shape, cfg))
        core = seed_labels == 1
        if cfg.method == "esrg" and cfg.purify_manual_seed:
            # SRG's seed_labels (used verbatim by grow_srg) is left untouched --
            # this only reshapes the ESRG growth core.
            core = seedmod.purify_core(core, cfg)
            s_info["core_area"] = int(core.sum())
        cand = core
    else:
        (core, s_info), dt = timed(lambda: seedmod.select_seed(img_masked, mask, depth, cfg))
        try:
            from skimage import filters as _f
            th = s_info.get("thresholds")
            cand = mask & (img_masked >= th[-1]) if th else np.zeros_like(mask)
        except Exception:
            cand = np.zeros_like(mask)

    stages.append(Stage("candidates", "4 · Candidates", cand, "mask",
                        {"thresholds": s_info.get("thresholds"),
                         "components found": s_info.get("n_raw"),
                         "components kept": s_info.get("n_kept"),
                         "detail": s_info.get("components", [])[:8]}, dt))
    stages.append(Stage("seed", "5 · Seed core", core, "mask",
                        {"status": s_info["status"], "chosen": s_info.get("chosen"),
                         "core area": s_info.get("core_area"),
                         **({"seed types": s_info["types"]} if s_info.get("types") else {}),
                         "warnings": s_info.get("warnings", [])}, 0.0,
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
    if cfg.method == "srg" and (cfg.seed_mode == "auto" or planted_core is not None):
        # No planted competitors: the auto core competes with a grid of
        # background seeds inside the head mask.
        (region, trace), dt = timed(lambda: growing.grow_srg_auto(img_masked, mask, core, cfg))
        growth_info = {"stop reason": trace["stop_reason"], "passes": trace["passes"],
                       "background seeds": trace["n_background_seeds"]}
    elif cfg.method == "srg":
        # Whole image, unmasked: the published algorithm has no skull-stripping step
        # and leaves no pixel unallocated. Every planted id grows under the same
        # rule; grow_srg has no notion of "tumor" at all.
        (label_map, trace), dt = timed(lambda: growing.grow_srg(img, seed_labels))
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
        (region, trace), dt = timed(lambda: growing.grow_esrg(L, mask, core, sigma_floor, cfg))
        growth_info = {"stop reason": trace["stop_reason"], "passes": trace["passes"]}
    stages.append(Stage("growth", "6 · Region growing", region, "mask", growth_info, dt,
                        "WARN" if "warning" in trace else "OK"))

    # ── Stage 7: post-processing + final ─────────────────────────────────────
    say("Post-processing…")
    (final, p_info), dt = timed(lambda: postprocess.postprocess(region, core, cfg))
    result.mask = final

    brain_frac = final.sum() / max(mask.sum(), 1)
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
