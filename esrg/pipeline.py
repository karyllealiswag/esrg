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


def run(image_path, cfg, mask_path=None, manual_points=None, progress=None):
    """
    Full pipeline on one slice. manual_points overrides seed selection when
    cfg.seed_mode == 'manual'. progress(str) is an optional GUI callback.
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
    img = pre.normalize(raw)
    meta = {"path": image_path, "shape": img.shape, "scale": round(scale, 3),
            "method": cfg.method, **parse_brisc_name(image_path)}
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
    say("Log transform…")
    L, dt = timed(lambda: pre.log_transform(img_masked, cfg.log_eps))
    sigma_floor = pre.noise_floor(L, mask, cfg.sigma_floor_min)
    stages.append(Stage("log", "3 · Log domain", L, "heat",
                        {"range": f"{L[mask].min():.2f}–{L[mask].max():.2f}",
                         "noise floor σ": round(sigma_floor, 4),
                         "note": "multiplicative bias becomes additive"}, dt))

    # ── Stage 4–5: seed selection ────────────────────────────────────────────
    say("Selecting seed…")
    if cfg.seed_mode == "manual":
        (core, s_info), dt = timed(lambda: seedmod.manual_seed(manual_points or [], img.shape, cfg))
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
    if cfg.method == "srg":
        (region, trace), dt = timed(lambda: growing.grow_srg(img_masked, mask, core, cfg))
    else:
        (region, trace), dt = timed(lambda: growing.grow_esrg(L, mask, core, sigma_floor, cfg))
    stages.append(Stage("growth", "6 · Region growing", region, "mask",
                        {"stop reason": trace["stop_reason"], "passes": trace["passes"],
                         **({"background seeds": trace["n_background_seeds"]}
                            if "n_background_seeds" in trace else {})}, dt))

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
    result.scores["seconds"] = result.meta.get("seconds")
    result.meta["ground_truth"] = mp
    return result
