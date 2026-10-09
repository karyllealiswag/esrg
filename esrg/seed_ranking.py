"""
seed_ranking.py — Phase 1: contrast/shape-aware candidate ranking.

Purpose : Choose among surviving candidates by what an enhancing tumor is — a
          compact region brighter than its surroundings — not by size alone.
Function : Scores each candidate on normalized local contrast (brightness minus a
          surrounding ring), interiority (inscribed radius), and a mild shape term
          (compactness, solidity); Frangi vesselness can veto tubular candidates.
          Returns the candidate list re-ranked best-first.
Notes   : Full-set evaluation showed contrast as a secondary weight (0.5) below
          interiority (1.0); vesselness gave no aggregate gain on 2D slices and is
          off by default. Kept as a transparent, inspectable scoring function.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


def vesselness_map(img, mask, sigmas):
    """Frangi vesselness in [0,1] inside the mask; high on tubular structures."""
    try:
        v = filters.frangi(img.astype(float), sigmas=sigmas, black_ridges=False)
    except Exception:
        return np.zeros_like(img, dtype=float)
    v = np.nan_to_num(v)
    m = v[mask]
    if m.size and m.max() > 0:
        v = v / m.max()
    v[~mask] = 0.0
    return np.clip(v, 0, 1)


def _contrast(comp, img, ring_width):
    """mean(component) minus mean(a ring just outside it). Positive = brighter than surround."""
    outer = morphology.binary_dilation(comp, morphology.disk(ring_width))
    ring = outer & ~comp
    if ring.sum() == 0:
        return 0.0
    return float(img[comp].mean() - img[ring].mean())


def _shape(comp):
    """Compactness and solidity in [0,1]; 1.0 = perfect disk / fully solid."""
    props = measure.regionprops(comp.astype(int))
    if not props:
        return 0.0, 0.0
    p = props[0]
    perim = p.perimeter if p.perimeter > 0 else 1.0
    compactness = min(1.0, 4 * np.pi * p.area / (perim * perim))
    solidity = float(p.solidity) if p.solidity == p.solidity else 0.0
    return float(compactness), solidity


def _vessel_frac(comp, vmap, thresh=0.5):
    """Share of component pixels that read as strongly tubular."""
    return float((vmap[comp] > thresh).mean()) if comp.any() else 0.0


def score_candidates(candidates, img, mask, cfg):
    """
    candidates : list of (record_dict, comp_bool, dt_float) as built in seeds.py
    Returns the same list re-ranked best-first, each record annotated with its
    feature values and final score. Purely additive: records gain keys, none lost.
    """
    if not candidates:
        return candidates

    vmap = (vesselness_map(img, mask, cfg.frangi_sigmas)
            if cfg.rank_use_vesselness else None)

    feats = []
    for rec, comp, dt in candidates:
        radius = float(dt.max())
        contrast = _contrast(comp, img, cfg.rank_contrast_ring) if cfg.rank_use_contrast else 0.0
        compactness, solidity = _shape(comp) if cfg.rank_use_shape else (0.0, 0.0)
        vfrac = _vessel_frac(comp, vmap) if vmap is not None else 0.0
        feats.append(dict(radius=radius, contrast=contrast,
                          compactness=compactness, solidity=solidity, vfrac=vfrac))

    # Normalize the size-like features across candidates so weights are comparable.
    def norm(key):
        vals = np.array([f[key] for f in feats], float)
        lo, hi = vals.min(), vals.max()
        return (vals - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(vals)

    n_radius, n_contrast = norm("radius"), norm("contrast")

    ranked = []
    for (rec, comp, dt), f, nr, nc in zip(candidates, feats, n_radius, n_contrast):
        shape_term = 0.5 * (f["compactness"] + f["solidity"]) if cfg.rank_use_shape else 0.0
        score = (cfg.rank_w_contrast * nc
                 + cfg.rank_w_radius * nr
                 + cfg.rank_w_shape * shape_term)
        # Vesselness is a veto, not a weight: a clearly tubular candidate is demoted hard.
        vetoed = cfg.rank_use_vesselness and f["vfrac"] > cfg.rank_vessel_max
        if vetoed:
            score -= 10.0
        rec.update(contrast=round(f["contrast"], 1), vfrac=round(f["vfrac"], 2),
                   compactness=round(f["compactness"], 2), solidity=round(f["solidity"], 2),
                   score=round(float(score), 3), vetoed=bool(vetoed),
                   # Exact Equation 3.3 terms, so the GUI can show Q = sum of them.
                   _terms={"contrast_n": float(nc), "radius_n": float(nr), "shape": float(shape_term),
                           "contrast": float(f["contrast"]), "radius": float(f["radius"]),
                           "score": float(score)})
        ranked.append((score, rec, comp, dt))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [(rec, comp, dt) for _, rec, comp, dt in ranked]
