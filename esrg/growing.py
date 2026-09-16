"""
growing.py — Phase C: region growing (ESRG and the SRG baseline).

Purpose : Grow the seed into the tumor region under the enhanced rules, and provide
          the unmodified Adams & Bischof (1994) algorithm as the control.
Function : grow_esrg() uses a sequentially sorted list keyed by a local log-domain
          difference (Objective 2) and absorbs a pixel only if it passes a local
          confidence bound and a global drift guard (Objective 3), with statistics
          frozen per pass. grow_srg() runs the classical multi-region tessellation
          with unconditional absorption over the seeds the user planted.
Notes   : Statistics use Welford's method; per-pass freezing prevents the positive
          feedback that causes leakage. A per-image adaptive bound is available
          (adaptive_k) but did not beat the fixed bound and is off by default.
"""
import heapq
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def grow_srg(img, seed_labels):
    """
    Adams & Bischof (1994) seeded region growing, exactly as published.

    seed_labels : integer seed map, one id per planted region (0 = unseeded).
                  Every region grows under the identical rule below -- no id is
                  special-cased in the loop. delta(x, region) = |img[x] -
                  mean(region)|, fixed at the moment x enters the SSL; the SSL
                  always pops the globally smallest delta next; absorption is
                  unconditional. The loop ends only when the SSL is empty, i.e.
                  every pixel has been allocated to some region.

    Runs on the whole image, not the head mask: the original has no
    skull-stripping step. Seed every distinct tissue you don't want merged
    together (including background/scalp), or those pixels get divided up
    among whichever regions happen to reach them first.

    Returns (label, trace). label is the full partition of img. Which id is
    "the structure of interest" is not decided here -- that is left to the
    caller, since SRG itself has no concept of a privileged region.
    """
    H, W = img.shape
    label = seed_labels.astype(np.int32).copy()

    ids = [int(i) for i in np.unique(label) if i > 0]
    sums = {i: float(img[label == i].sum()) for i in ids}
    counts = {i: int((label == i).sum()) for i in ids}

    ssl = []

    def enqueue(r, c, lid):
        """delta is computed once, here, against the region mean of the moment."""
        if 0 <= r < H and 0 <= c < W and label[r, c] == 0:
            heapq.heappush(ssl, (abs(img[r, c] - sums[lid] / counts[lid]), r, c, lid))

    for r, c in zip(*np.nonzero(label)):
        for dr, dc in NEIGH8:
            enqueue(r + dr, c + dc, int(label[r, c]))

    while ssl:
        _, r, c, lid = heapq.heappop(ssl)
        if label[r, c] != 0:
            continue                    # a better-fitting region reached it first
        label[r, c] = lid               # unconditional: no stopping rule, no re-scoring
        sums[lid] += float(img[r, c])
        counts[lid] += 1
        for dr, dc in NEIGH8:
            enqueue(r + dr, c + dc, lid)

    trace = {
        "stop_reason": "tessellation complete (no stopping criterion)",
        "n_seed_regions": len(ids),
        "unlabeled": int((label == 0).sum()),
        "region_areas": {i: int(counts[i]) for i in ids},
        "region_means": {i: round(sums[i] / counts[i], 2) for i in ids},
    }
    if len(ids) < 2:
        trace["warning"] = ("Only one seed region was planted. SRG has no stopping "
                            "rule, so with nothing to compete against, that single "
                            "region absorbs the entire image.")
    return label, trace
