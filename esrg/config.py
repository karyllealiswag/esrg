"""
config.py — Central configuration for the ESRG pipeline.

Purpose : Hold every tunable parameter in one dataclass so runs are reproducible
          and the ablation switches (Chapter 3, Table 3.3) live in one place.
Function : `Config` is a frozen-by-convention dataclass; `.replace(**kw)` returns a
          validated copy for experiments, and `.save`/`.load` persist it as JSON.
Notes   : Values reflect the final tuning frozen on the TRAIN split. Phase-1/2/3
          research switches (contrast ranking, orbit exclusion, adaptive k) default
          to their evaluated settings; changing them re-opens the ablation.
"""
from dataclasses import dataclass, asdict, field
import json


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

    # ── Input ─────────────────────────────────────────────────────────
    max_side: int = 512             # longer side is resized to this (keeps runtime bounded)

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
    manual_seed_types: int = 4      # number of seed types selectable in manual mode
                                    # (type 1 = tumor, 2+ = competing regions for SRG)
    purify_manual_seed: bool = True # ESRG + manual seeding only: erode the click disk to
                                    # its medial pixels (same seed_core_frac test as the
                                    # auto core) before growing, so a boundary/partial-
                                    # volume pixel the disk happened to catch can't inflate
                                    # the initial sigma_A the stopping bound is based on.
                                    # No effect on seed_mode="auto" or on method="srg".

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
    max_passes: int = 4             # P_max
    sigma_floor_min: float = 0.02   # lower clamp for the estimated noise floor (log units)
    lazy_tol: float = 1e-9          # re-queue tolerance for lazy re-evaluation

    # ── Baseline SRG ──────────────────────────────────────────────────
    bg_seed_step: int = 8           # grid step for automatic background seeds
                                    # (SRG + automatic seeding only)

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
        return self

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f)).validate()

    def replace(self, **kw):
        """Copy with overrides (used by ablation / tuning scripts)."""
        d = self.to_dict()
        d.update(kw)
        return Config(**d).validate()
