"""
config.py — Central configuration for the ESRG pipeline.

Purpose : Hold every tunable parameter in one dataclass so runs are reproducible
          and the ablation switches (Chapter 3, Table 3.3) live in one place.
Function : `Config` is a frozen-by-convention dataclass; `.replace(**kw)` returns a
          validated copy for experiments, and `.save`/`.load` persist it as JSON.
Notes   : Defaults are the values frozen on the BRISC TRAIN split. BraTS runs load
          configs/esrg_brats_frozen.json (pre_stripped=True, tuned on the BraTS
          development set, experiments/PROTOCOL_BRATS.md) instead. Phase-1/2/3
          research switches (contrast ranking, orbit exclusion, adaptive k) default
          to their evaluated settings; changing them re-opens the ablation.
"""
from dataclasses import dataclass, asdict, field
import json


REMOVED = ("purify_manual_seed", "bg_seed_step")


@dataclass
class Config:
    # ── Method selection ──────────────────────────────────────────────
    method: str = "esrg"            # "esrg" (proposed) | "srg" (Adams & Bischof baseline)
    seed_mode: str = "auto"         # "auto" (Objective 1) | "manual" (user clicks)

    # ── Ablation switches (Chapter 3, Experiment E4) ──────────────────
    use_log: bool = True            # Objective 2: L(x) = ln(I(x) + eps); off -> L(x) = I(x)
    use_log_local: bool = True      # Objective 2: local log-domain difference measure
    use_stopping: bool = True       # Objective 3: adaptive stopping criterion
    use_n4: bool = False            # N4 bias correction (ablation arm only, never part of ESRG)
    esrg_compete: bool = False      # ESRG + manual/planted seeding only: seeds of type >= 2
                                    # grow as SRG competitors over the whole image (Experiment
                                    # E8). Off = ESRG as proposed, which ignores them.

    # ── Input ─────────────────────────────────────────────────────────
    max_side: int = 512             # longer side is resized to this (keeps runtime bounded)
    pre_stripped: bool = False      # input is already skull-stripped (BraTS): the head
                                    # mask is the non-zero brain, no skull stripping

    # ── Skull stripping ───────────────────────────────────────────────
    frame_open_radius: int = 2      # removes thin frames / text / arrows before head detection
    ss_max_radius: int = 12         # upper bound of the granulometric erosion search
    ss_rim_band: int = 4            # width (px) of the head rim used to detect scalp contact
    ss_min_brain_frac: float = 0.25 # brain must keep >= this fraction of the head area

    # ── Seed selection (Objective 1) ──────────────────────────────────
    otsu_classes: int = 3           # K
    seed_core_frac: float = 0.7     # alpha
    min_cand_area: int = 40         # A_min (pixels, at 512 px scale)
    interior_frac: float = 0.12     # histogram uses pixels deeper than this share of
                                    # the max head depth (excludes scalp and skull)
    min_clearance: float = 8.0      # a candidate's core must sit this far inside the
                                    # head outline, beyond its own inscribed radius
    manual_seed_radius: int = 3     # manual clicks are dilated into a small region
    manual_seed_types: int = 4      # seed types: 1 = tumor, 2 = healthy tissue,
                                    # 3 = background (air), 4 = scalp/skull (heads that
                                    # are not skull-stripped); 2-4 compete with the tumor

    # ── Phase 1: contrast/shape-aware seed ranking (Improvement Masterplan) ───
    rank_use_contrast: bool = True  # rank by brightness relative to local surround
    rank_use_vesselness: bool = False # off by default: no measurable gain on 2D
                                    # brain slices and ~2x slower; kept for ablation
    rank_use_shape: bool = True     # mild compactness/solidity tie-break
    rank_contrast_ring: int = 6     # width (px) of the surround ring for contrast
    rank_w_contrast: float = 0.5    # weight of the contrast term (tuned on train)
    rank_w_radius: float = 1.0      # weight of the interiority (inscribed-radius) term
    rank_w_shape: float = 0.2       # weight of the shape term (kept mild for gliomas)
    rank_vessel_max: float = 0.6    # candidate rejected if vesselness fraction exceeds this
    frangi_sigmas: tuple = (1.0, 2.0, 3.0)  # scales for the Frangi vesselness map

    # ── Phase 2a: anatomical exclusion (orbits, skull base) ───────────────────
    exclude_orbits: bool = False    # ablation only: no aggregate gain (orbits are
                                    # dark on most slices and already excluded by Otsu)
    exclude_skull_base: bool = False# narrow skull-base band (off: minor bucket-C share)
    orbit_bright_pct: float = 88.0  # intensity percentile defining "bright" tissue
    orbit_min_area: int = 60        # eyeball area range (px, at 512 scale)
    orbit_max_area: int = 4000
    orbit_min_circularity: float = 0.65  # eyeballs are round; tumors often are not
    orbit_max_rel_y: float = 0.40   # orbits sit in the upper (anterior) head region
    skullbase_band_frac: float = 0.12    # bottom fraction of head treated as skull base
    skullbase_depth: float = 12.0   # only shallow floor tissue, not deep central tumor

    # ── Phase 3: per-image adaptive stopping bound (Improvement Masterplan) ────
    adaptive_k: bool = False        # ablation only: did not beat a well-chosen fixed
                                    # k_L on meningioma; kept for completeness
    adaptive_k_lo: float = 1.3      # k_L for a low-contrast tumor (tighter, avoids bleed)
    adaptive_k_hi: float = 2.0      # k_L for a high-contrast tumor (looser, reaches edge)
    adaptive_ring: int = 5          # px ring around the seed used to measure contrast
    adaptive_ref_contrast: float = 0.5  # log-domain contrast that maps to the high end

    # ── Region growing (Objectives 2 & 3) ─────────────────────────────
    log_eps: float = 1.0            # epsilon in ln(I + eps)
    local_radius: int = 3           # r  -> (2r+1)x(2r+1) window
    k_local: float = 1.5            # k_L
    k_global: float = 3.0           # k_G  (must be > k_L)
    sigma_cap: float = None         # upper clamp on sigma_A: min(max(s_A, floor), max(cap, floor)).
                                    # None = off (the proposed method as evaluated so far)
    flood_frac: float = None        # stopping-bound arms: if the grown region exceeds this
                                    # fraction of the brain mask, regrow with k_L, k_G * flood_shrink
    flood_shrink: float = 0.7       # per-retry multiplier of k_L and k_G (cumulative)
    flood_tries: int = 3            # maximum retries
    global_running: bool = False    # ASC global test against the running region mean (per
                                    # absorbed pixel) instead of the mean frozen per pass
    healthy_rule: str = "off"       # healthy-tissue rule: "off" | "veto" (reject pixels nearer the
                                    # healthy prototype than the region mean) | "rescue" (absorb a
                                    # pixel that fails only the local bound unless it is nearer the
                                    # healthy prototype) | "both"
    healthy_proto: str = "median"   # healthy prototype over the brain interior: "median" | "mode"
                                    # (histogram peak) | "otsu3" (middle class of a 3-class Otsu)
    max_passes: int = 4             # P_max
    sigma_floor_min: float = 0.02   # lower clamp for the estimated noise floor (log units)
    lazy_tol: float = 1e-9          # re-queue tolerance for lazy re-evaluation

    # ── Baseline SRG ──────────────────────────────────────────────────
    # With automatic seeding the tumor seed is selected automatically and the
    # competing seeds (types 2-3) are planted, exactly as under manual seeding.
    # ESRG modules grafted onto the baseline, one at a time, to measure what each
    # adds to SRG on its own (GUI "Applied Methods"). Off = the 1994 algorithm, so
    # every batch configuration dict(method="srg") is unchanged. They act on the
    # tumor region (region 1) only; background regions keep the original rule.
    srg_use_log_local: bool = False # Objective 2: grow on L = ln(I + eps) with the
                                    # local measure delta = |L - mu_loc| (radius local_radius)
    srg_use_stopping: bool = False  # Objective 3: region 1 absorbs only within
                                    # T_L = k_local*sigma_A and T_G = k_global*sigma_A
    # The modules' own tuned values (BraTS: tuned on the same grid as ESRG, PROTOCOL_BRATS
    # Section 5.3). None = share ESRG's local_radius / k_local / k_global (BRISC behaviour).
    srg_use_healthy: bool = False   # the healthy-tissue rule on region 1, with ESRG's healthy_rule
    srg_connectivity: int = 8       # 8 as in Adams & Bischof (1994), Sec. II; 4 reproduces results
                                    # computed before 2026-10-13
    srg_local_radius: int = None
    srg_k_local: float = None
    srg_k_global: float = None

    # ── Post-processing ───────────────────────────────────────────────
    post_open_radius: int = 1

    # ── Diagnostics ───────────────────────────────────────────────────
    leak_warn_frac: float = 0.40    # warn if tumor mask > this fraction of the brain
    leak_ratio: float = 2.0         # Leakage Rate: predicted area > leak_ratio * GT area

    def validate(self):
        """Fail fast on impossible settings."""
        assert self.method in ("esrg", "srg"), "method must be 'esrg' or 'srg'"
        assert self.seed_mode in ("auto", "manual"), "seed_mode must be 'auto' or 'manual'"
        assert self.otsu_classes >= 2, "otsu_classes must be >= 2"
        assert 0 < self.seed_core_frac <= 1, "seed_core_frac must be in (0, 1]"
        assert self.k_global > self.k_local, "k_global must exceed k_local"
        assert self.local_radius >= 1 and self.max_passes >= 1
        assert self.manual_seed_types >= 1, "manual_seed_types must be >= 1"
        assert self.srg_k("global") > self.srg_k("local"), "srg_k_global must exceed srg_k_local"
        assert self.sigma_cap is None or self.sigma_cap > 0, "sigma_cap must be positive"
        assert self.healthy_rule in ("off", "veto", "rescue", "both"), "healthy_rule: off|veto|rescue|both"
        assert self.healthy_proto in ("median", "mode", "otsu3"), "healthy_proto: median|mode|otsu3"
        assert self.srg_connectivity in (4, 8), "srg_connectivity must be 4 or 8"
        assert not (self.srg_use_healthy and self.healthy_rule == "off"),             "srg_use_healthy mirrors ESRG's healthy_rule, which must not be 'off'"
        assert self.flood_frac is None or 0 < self.flood_frac <= 1, "flood_frac must be in (0, 1]"
        assert 0 < self.flood_shrink < 1 and self.flood_tries >= 1
        return self

    def srg_k(self, which):
        """k_L ('local') or k_G ('global') read by the SRG + ASC module."""
        own = self.srg_k_local if which == "local" else self.srg_k_global
        return own if own is not None else (self.k_local if which == "local" else self.k_global)

    def srg_radius(self):
        """Window radius read by the SRG + LLDM module."""
        return self.srg_local_radius if self.srg_local_radius is not None else self.local_radius

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        """Read a saved config; keys of settings since removed (purify_manual_seed,
        bg_seed_step) are dropped so older files still load."""
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k not in REMOVED}).validate()

    def replace(self, **kw):
        """Copy with overrides (used by ablation / tuning scripts)."""
        d = self.to_dict()
        d.update(kw)
        return Config(**d).validate()
