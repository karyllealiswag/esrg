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
          seeding it lists neither the head mask nor the candidates. Seed types:
          1 tumor, 2 healthy tissue, 3 background, 4 scalp/skull; the competing
          types are planted the same way under automatic and manual seeding.
"""
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import preprocessing as pre
from . import seeds as seedmod
from . import growing, postprocess, metrics
from .io_utils import load_image, load_mask, find_mask_for, parse_name


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
    Full pipeline on one slice. progress(str) is an optional GUI callback.

    manual_points : [(row, col, type), ...]; type 1 tumor, 2 healthy tissue,
                    3 background (air), 4 scalp/skull. Under cfg.seed_mode == 'manual' every type is
                    used. Under automatic seeding the tumor seed is selected
                    automatically and only types >= 2 are read: they are the
                    competing seeds of SRG (and of ESRG in compete mode), planted
                    the same way in both seeding modes.

    Evaluation-only hooks (experiments/evaluate.py):
      planted_core : boolean tumor seed that replaces seed selection; competing
                     seeds still come from manual_points (types >= 2).
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
            **parse_name(image_path)}
    stages.append(Stage("input", "1 · Input", img, "gray",
                        {"size": f"{img.shape[1]} × {img.shape[0]} px",
                         "scale factor": round(scale, 3), "file": image_path,
                         "_norm": norm_info, **({"_raw": raw} if record else {})}, dt))

    # ── Stage 2: head mask (brain mask of a skull-stripped slice) ────────────
    say("Masking head…")
    (mask, depth, ss_info), dt = timed(lambda: pre.head_mask(img, cfg))
    srg = cfg.method == "srg"
    manual = cfg.seed_mode == "manual" and planted_core is None
    srg_manual = srg and manual
    # SRG with manual seeding never reads the head mask to seed or grow (the 1994
    # algorithm has no skull stripping), so it is not one of its stages; it is still
    # computed for the noise floor and the |M|/|H| leakage warning.
    if not srg_manual:
        stages.append(Stage("mask", "Brain mask" if cfg.pre_stripped else "Head mask",
                            mask, "mask", ss_info, dt, "WARN" if ss_info["warnings"] else "OK"))
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

    # Competing seeds (types >= 2), planted identically under either seeding mode.
    comp_points = [p for p in (manual_points or []) if len(p) > 2 and int(p[2]) >= 2]
    comp = seedmod.manual_seed(comp_points, img.shape, cfg)[0] if comp_points else np.zeros(img.shape, np.int32)
    # SRG tessellates the whole image; so does ESRG in compete mode once competitors exist.
    whole = srg or (cfg.esrg_compete and comp.any())

    # ── Stage 3: log domain ──────────────────────────────────────────────────
    # ESRG always lists it (with the transform off it still yields the noise floor).
    # SRG grows on intensity unless the Objective 2 module is applied, and needs the
    # noise floor only for the Objective 3 module; it lists Stage 3 only when one of
    # them is on. The noise floor is always measured inside H, so SRG + ASC and ESRG
    # read the same tissue-noise estimate (the air outside H would drive it to zero).
    base = img if whole else img_masked
    if srg:
        use_log, show = cfg.srg_use_log_local, cfg.srg_use_log_local or cfg.srg_use_stopping
    else:
        use_log, show = cfg.use_log, True
    if use_log:
        say("Log transform…")
        L, dt = timed(lambda: pre.log_transform(base, cfg.log_eps))
    else:
        # Ablation: growth and the noise floor run on the normalized intensity itself.
        L, dt = base.copy(), 0.0
    nf = pre.noise_floor_details(L, mask, cfg.sigma_floor_min) if show else None
    sigma_floor = nf["sigma_floor"] if nf else None
    if show:
        name = ("2 · Log domain" if use_log else "2 · Noise floor" if srg else "2 · Log domain (off)")
        stages.append(Stage("log", name, L, "heat",
                            {"transform": f"L = ln(I + {cfg.log_eps:g})" if use_log
                                          else "off — L = I (no log transform)",
                             "range": f"{L[mask].min():.2f}–{L[mask].max():.2f}",
                             "noise floor σ": round(sigma_floor, 4),
                             "note": "multiplicative bias becomes additive" if use_log
                                     else "multiplicative bias stays multiplicative",
                             "_use_log": use_log, "_eps": cfg.log_eps, "_noise_floor": nf,
                             "_region": mask, "_base": base}, dt))

    # ── Stage 4–5: seed selection ────────────────────────────────────────────
    say("Selecting seed…")
    if planted_core is not None:
        core = planted_core.astype(bool).copy()
        s_info = {"status": "OK" if core.any() else "NO SEED", "core_area": int(core.sum()),
                  "components": [], "warnings": []}
        cand, dt = core, 0.0
    elif manual:
        (seed_labels, s_info), dt = timed(lambda: seedmod.manual_seed(manual_points or [], img.shape, cfg))
        core = seed_labels == 1
        cand = core
    else:
        (core, s_info), dt = timed(lambda: seedmod.select_seed(img_masked, mask, depth, cfg))
        try:
            th = s_info.get("thresholds")
            cand = mask & (img_masked >= th[-1]) if th else np.zeros_like(mask)
        except Exception:
            cand = np.zeros_like(mask)
    if not manual and comp.any():
        s_info["types"] = {"type 1 (auto)" if planted_core is None else "type 1 (planted)": int(core.sum()),
                           **{f"type {t}": int((comp == t).sum()) for t in np.unique(comp) if t >= 2}}
    # Full seed map: the tumor seed as id 1 over the competing seeds.
    seed_labels = np.where(core, 1, comp).astype(np.int32)

    if not srg_manual:  # SRG uses the clicks as they are: no candidate step
        stages.append(Stage("candidates", "3 · Candidates", cand, "mask",
                            {"thresholds": s_info.get("thresholds"),
                             "components found": s_info.get("n_raw"),
                             "components kept": s_info.get("n_kept"),
                             "detail": s_info.get("components", [])[:8], "_s_info": s_info}, dt))
    stages.append(Stage("seed", "4 · Seed core", core, "mask",
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
        stages.append(Stage("final", "6 · Final", result.mask, "overlay",
                            {"status": result.status}, 0.0, "FAIL"))
        meta["seconds"] = round(time.perf_counter() - t_total, 3)
        return _attach_scores(result, mask_path, image_path, cfg, core)

    # ── Stage 6: region growing ──────────────────────────────────────────────
    say("Growing region…")

    # Healthy-tissue prototype (cfg.healthy_rule; SRG mirrors it with cfg.srg_use_healthy):
    # a robust centre of L over the brain interior, the same region seed selection reads.
    mu_H = None
    if (cfg.srg_use_healthy if srg else cfg.healthy_rule != "off"):
        interior = mask & (depth >= max(cfg.min_clearance, cfg.interior_frac * depth.max()))
        mu_H = pre.healthy_prototype(L, interior if interior.sum() >= 50 else mask, cfg.healthy_proto)
    srg_healthy = dict(mu=mu_H, rule=cfg.healthy_rule) if (srg and cfg.srg_use_healthy) else None

    def grow(c):
        """One growth attempt under settings c: (region, trace, growth_info, tessellation stage or None)."""
        # Applied ESRG modules for the SRG baseline (region 1 only; both None = 1994 SRG).
        srg_local = c.srg_radius() if c.srg_use_log_local else None
        srg_stop = (dict(k_L=c.srg_k("local"), k_G=c.srg_k("global"), sigma_floor=sigma_floor,
                         sigma_cap=c.sigma_cap) if c.srg_use_stopping else None)
        if srg:
            # Whole image, unmasked: the published algorithm has no skull-stripping step
            # and leaves no pixel unallocated. Every seed id grows under the same rule;
            # grow_srg has no notion of "tumor" at all. Under automatic seeding id 1 is
            # the auto core, under manual seeding the operator's tumor click.
            (label_map, tr), t = timed(lambda: growing.grow_srg(L, seed_labels, record, srg_local, srg_stop,
                                                                c.srg_connectivity, srg_healthy))
            tess = Stage("tessellation", "6a · Full tessellation", label_map, "labels",
                         {"seed regions": tr["n_seed_regions"], "region areas": tr["region_areas"],
                          "region means": tr["region_means"]}, 0.0)
            info = {"stop reason": tr["stop_reason"], "seed regions": tr["n_seed_regions"],
                    "unlabeled px": tr["unlabeled"], **({"warning": tr["warning"]} if "warning" in tr else {})}
            return label_map == 1, tr, info, tess, t
        if whole:
            # ESRG compete mode (Experiment E8): the tumor grows under the ESRG rules,
            # the planted competitors under the 1994 rule, over the whole image as SRG.
            (reg, tr), t = timed(lambda: growing.grow_esrg(L, np.ones_like(mask), core, sigma_floor,
                                                           c, record, competitors=comp, healthy_mu=mu_H))
            tess = Stage("tessellation", "6a · Competing regions", tr["labels"], "labels",
                         {"seed regions": tr["n_seed_regions"], "region areas": tr["region_areas"],
                          "region means": tr["region_means"]}, 0.0)
            info = {"stop reason": tr["stop_reason"], "passes": tr["passes"],
                    "seed regions": tr["n_seed_regions"], "unlabeled px": tr["unlabeled"]}
            return reg, tr, info, tess, t
        (reg, tr), t = timed(lambda: growing.grow_esrg(L, mask, core, sigma_floor, c, record, healthy_mu=mu_H))
        return reg, tr, {"stop reason": tr["stop_reason"], "passes": tr["passes"]}, None, t

    region, trace, growth_info, tess, dt = grow(cfg)
    # Flood guard (cfg.flood_frac; off by default): a stopping-bound arm whose region covers
    # more than flood_frac of the brain is regrown with k_L and k_G shrunk at each retry.
    bounded = cfg.srg_use_stopping if srg else cfg.use_stopping
    retries = 0
    if cfg.flood_frac and bounded:
        while (region & mask).sum() > cfg.flood_frac * mask.sum() and retries < cfg.flood_tries:
            retries += 1
            f = cfg.flood_shrink ** retries
            if srg:
                kl, kg = cfg.srg_k("local") * f, cfg.srg_k("global") * f
                retry_cfg = cfg.replace(srg_k_local=kl, srg_k_global=kg)
            else:
                retry_cfg = cfg.replace(k_local=cfg.k_local * f, k_global=cfg.k_global * f)
            region, trace, growth_info, tess, dt2 = grow(retry_cfg)
            dt += dt2
        if retries:
            trace["flood_retries"] = retries
            growth_info["flood guard retries"] = retries
    if mu_H is not None:
        growth_info["healthy prototype"] = round(mu_H, 4)
    if tess is not None:
        stages.append(tess)
    growth_info["_trace"] = trace
    stages.append(Stage("growth", "5 · Region growing", region, "mask", growth_info, dt,
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
    stages.append(Stage("final", "6 · Final", final, "overlay", p_info, dt, status))

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
    result.scores["mask_sha1"] = hashlib.sha1(np.packbits(result.mask.astype(bool))).hexdigest()[:16]
    result.meta["ground_truth"] = mp
    return result
