"""
growing.py — Phase C: region growing (ESRG and the SRG baseline).

Purpose : Grow the seed into the tumor region under the enhanced rules, and provide
          the unmodified Adams & Bischof (1994) algorithm as the control.
Function : grow_esrg() uses a sequentially sorted list keyed by a local log-domain
          difference (Objective 2) and absorbs a pixel only if it passes a local
          confidence bound and a global drift guard (Objective 3), with statistics
          frozen per pass. grow_srg() runs the classical tumor-vs-background
          tessellation with unconditional absorption.
Notes   : Statistics use Welford's method; per-pass freezing prevents the positive
          feedback that causes leakage. A per-image adaptive bound is available
          (adaptive_k) but did not beat the fixed bound and is off by default.
"""
import heapq
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class _LocalMean:
    """Running mean of absorbed log-intensities inside a (2r+1)^2 window."""

    def __init__(self, shape, radius):
        self.s = np.zeros(shape)          # sum of L over absorbed pixels
        self.n = np.zeros(shape)          # count of absorbed pixels
        self.r = radius
        self.H, self.W = shape

    def add(self, r, c, value):
        self.s[r, c] = value
        self.n[r, c] = 1

    def mean(self, r, c):
        """None when the window holds no absorbed pixel yet."""
        r0, r1 = max(0, r - self.r), min(self.H, r + self.r + 1)
        c0, c1 = max(0, c - self.r), min(self.W, c + self.r + 1)
        cnt = self.n[r0:r1, c0:c1].sum()
        if cnt == 0:
            return None
        return self.s[r0:r1, c0:c1].sum() / cnt


class _Welford:
    """Online mean and variance (Welford, 1962); read only at pass boundaries."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def sigma(self):
        return float(np.sqrt(self.m2 / (self.n - 1))) if self.n > 1 else 0.0


def grow_esrg(L, mask, core, sigma_floor, cfg):
    """
    L           : log-domain image
    mask        : head mask (growth never leaves it)
    core        : seed core
    sigma_floor : noise floor, lower clamp on sigma_A
    """
    H, W = L.shape
    region = core.copy()
    rejected = np.zeros((H, W), bool)
    trace = {"passes": [], "stop_reason": None}

    local = _LocalMean(L.shape, cfg.local_radius)
    stats = _Welford()
    for r, c in zip(*np.nonzero(core)):
        local.add(r, c, L[r, c])
        stats.add(L[r, c])

    def delta(r, c):
        """Objective 2: compare against nearby absorbed pixels, not the whole region."""
        if not cfg.use_log_local:
            return abs(L[r, c] - stats.mean)      # ablation: global reference
        m = local.mean(r, c)
        return abs(L[r, c] - (stats.mean if m is None else m))

    # Phase 3: set the acceptance multiplier k_L per image from how well the seed
    # stands out from its immediate surroundings (log-domain contrast). A tumor
    # that is clearly brighter than adjacent tissue can grow to a sharp edge with
    # a looser bound; a low-contrast tumor needs a tighter bound to avoid bleeding
    # into similar-intensity neighbours. k_G (drift ceiling) stays fixed.
    if cfg.adaptive_k:
        seed_mean = float(np.mean([L[r, c] for r, c in zip(*np.nonzero(core))]))
        ring = ndi.binary_dilation(core, morphology.disk(cfg.adaptive_ring)) & ~core & mask
        if ring.any():
            contrast = abs(seed_mean - float(L[ring].mean()))
            t = min(1.0, contrast / cfg.adaptive_ref_contrast)   # 0..1
            k_local = cfg.adaptive_k_lo + t * (cfg.adaptive_k_hi - cfg.adaptive_k_lo)
        else:
            k_local = cfg.k_local
        trace["adaptive_k_local"] = round(float(k_local), 3)
    else:
        k_local = cfg.k_local

    for p in range(1, cfg.max_passes + 1):
        # Statistics frozen for the whole pass: updating per pixel lets each
        # borderline absorption widen sigma and admit the next one (leakage).
        mu_A = stats.mean
        sigma_A = max(stats.sigma(), sigma_floor)
        t_local = k_local * sigma_A
        t_global = cfg.k_global * sigma_A

        ssl, queued = [], np.zeros((H, W), bool)
        for r, c in zip(*np.nonzero(region)):
            for dr, dc in NEIGH8:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and mask[nr, nc]
                        and not region[nr, nc] and not rejected[nr, nc]
                        and not queued[nr, nc]):
                    heapq.heappush(ssl, (delta(nr, nc), nr, nc))
                    queued[nr, nc] = True

        added, stop = 0, "queue exhausted"
        while ssl:
            d, r, c = heapq.heappop(ssl)
            if region[r, c] or rejected[r, c]:
                continue

            # Lazy re-evaluation: the local mean may have moved since queueing.
            d_now = delta(r, c)
            if d_now > d + cfg.lazy_tol:
                heapq.heappush(ssl, (d_now, r, c))
                continue

            if cfg.use_stopping:
                # The SSL is sorted, so the first failure implies all the rest fail.
                if d_now > t_local:
                    stop = f"delta {d_now:.3f} > T_L {t_local:.3f}"
                    break
                # Global drift guard: blocks slow chaining across a soft boundary.
                if abs(L[r, c] - mu_A) > t_global:
                    rejected[r, c] = True
                    continue

            region[r, c] = True
            local.add(r, c, L[r, c])
            stats.add(L[r, c])
            added += 1
            for dr, dc in NEIGH8:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and mask[nr, nc]
                        and not region[nr, nc] and not rejected[nr, nc]
                        and not queued[nr, nc]):
                    heapq.heappush(ssl, (delta(nr, nc), nr, nc))
                    queued[nr, nc] = True

        trace["passes"].append({
            "pass": p, "mu": round(mu_A, 4), "sigma": round(sigma_A, 4),
            "T_L": round(t_local, 4), "T_G": round(t_global, 4),
            "added": added, "area": int(region.sum()), "stop": stop})

        if added == 0:
            trace["stop_reason"] = f"converged after pass {p} (no pixels added)"
            break
    else:
        trace["stop_reason"] = f"reached P_max = {cfg.max_passes}"
    if trace["stop_reason"] is None:
        trace["stop_reason"] = f"converged after pass {p}"

    return region, trace


def grow_srg(img, mask, core, cfg):
    """
    Control group: Adams & Bischof (1994) exactly as published.

    Tumor seeds compete with a grid of background seeds, delta is the raw
    distance to the region mean, and absorption is unconditional -- the
    algorithm runs until every pixel in the mask is allocated.
    """
    H, W = img.shape
    label = np.zeros((H, W), np.int32)     # 0 unlabeled, 1 tumor, >=2 background
    label[core] = 1

    sums = {1: float(img[core].sum())}
    counts = {1: int(core.sum())}

    # Background seeds on a regular grid, skipping the tumor core and its margin
    guard = ndi.binary_dilation(core, ndi.generate_binary_structure(2, 2), iterations=6)
    nxt = 2
    for r in range(0, H, cfg.bg_seed_step):
        for c in range(0, W, cfg.bg_seed_step):
            if mask[r, c] and not guard[r, c] and label[r, c] == 0:
                label[r, c] = nxt
                sums[nxt] = float(img[r, c])
                counts[nxt] = 1
                nxt += 1

    ssl, queued = [], np.zeros((H, W), bool)

    def enqueue(r, c):
        if (0 <= r < H and 0 <= c < W and mask[r, c]
                and label[r, c] == 0 and not queued[r, c]):
            best = None
            for dr, dc in NEIGH4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and label[nr, nc] > 0:
                    lid = label[nr, nc]
                    d = abs(img[r, c] - sums[lid] / counts[lid])
                    if best is None or d < best[0]:
                        best = (d, lid)
            if best:
                heapq.heappush(ssl, (best[0], r, c, best[1]))
                queued[r, c] = True

    for r, c in zip(*np.nonzero(label)):
        for dr, dc in NEIGH4:
            enqueue(r + dr, c + dc)

    while ssl:
        d, r, c, lid = heapq.heappop(ssl)
        if label[r, c] != 0:
            continue
        # Re-evaluate against the current neighbourhood before absorbing
        best = None
        for dr, dc in NEIGH4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and label[nr, nc] > 0:
                l2 = label[nr, nc]
                d2 = abs(img[r, c] - sums[l2] / counts[l2])
                if best is None or d2 < best[0]:
                    best = (d2, l2)
        if best is None:
            queued[r, c] = False
            continue
        lid = best[1]
        label[r, c] = lid                       # unconditional: no stopping rule
        sums[lid] += float(img[r, c])
        counts[lid] += 1
        for dr, dc in NEIGH4:
            enqueue(r + dr, c + dc)

    region = label == 1
    trace = {"passes": [{"pass": 1, "mu": round(sums[1] / counts[1], 2),
                         "sigma": None, "T_L": None, "T_G": None,
                         "added": int(region.sum() - core.sum()),
                         "area": int(region.sum()),
                         "stop": "all pixels allocated"}],
             "stop_reason": "tessellation complete (no stopping criterion)",
             "n_background_seeds": nxt - 2}
    return region, trace
