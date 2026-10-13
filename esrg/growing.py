"""
growing.py — Phase C: region growing (ESRG and the SRG baseline).

Purpose : Grow the seed into the tumor region under the enhanced rules, and provide
          the unmodified Adams & Bischof (1994) algorithm as the control.
Function : grow_esrg() uses a sequentially sorted list keyed by a local log-domain
          difference (Objective 2) and absorbs a pixel only if it passes a local
          confidence bound and a global drift guard (Objective 3), with statistics
          frozen per pass; given competing seeds (cfg.esrg_compete, Experiment E8) it
          grows the tumor under those rules while the competitors grow under the 1994
          rule. grow_srg() runs the classical multi-region tessellation with
          unconditional absorption over the planted seeds (8-connected, as published); under
          automatic seeding the tumor seed is the auto core and the competing seeds
          are still planted, so both seeding modes use it.
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


def grow_esrg(L, mask, core, sigma_floor, cfg, record=False, competitors=None, healthy_mu=None):
    """
    L           : log-domain image
    mask        : head mask (growth never leaves it)
    core        : seed core
    sigma_floor : noise floor, lower clamp on sigma_A
    record      : GUI only. Logs every decision in trace["events"] and the frozen
                  statistics of each pass in trace["pass_detail"]; the arithmetic
                  and the grown region are identical either way.
    competitors : compete mode only (cfg.esrg_compete, Experiment E8): integer map
                  of the planted competing regions (ids >= 2, 0 elsewhere). None, or
                  no competitor pixel, runs ESRG as proposed.
    healthy_mu  : healthy prototype for cfg.healthy_rule (median of the brain interior);
                  None or rule "off" leaves ESRG as before.
    """
    if competitors is not None and (competitors >= 2).any():
        return _grow_esrg_compete(L, mask, core, sigma_floor, cfg, competitors, record, healthy_mu)
    H, W = L.shape
    region = core.copy()
    rejected = np.zeros((H, W), bool)
    trace = {"passes": [], "stop_reason": None}
    events, pass_detail = ([], []) if record else (None, None)

    local = _LocalMean(L.shape, cfg.local_radius)
    stats = _Welford()
    for r, c in zip(*np.nonzero(core)):
        local.add(r, c, L[r, c])
        stats.add(L[r, c])
    if record:
        trace["seed_stats"] = {"n": stats.n, "mean": stats.mean, "s": stats.sigma()}

    def reference(r, c):
        """The value x is compared with: mu_loc(x) (Eq. 3.8), or mu_A when the window is empty."""
        if not cfg.use_log_local:
            return stats.mean                     # ablation: global reference
        m = local.mean(r, c)
        return stats.mean if m is None else m

    def delta(r, c):
        """Objective 2: compare against nearby absorbed pixels, not the whole region."""
        return abs(L[r, c] - reference(r, c))

    # Phase 3: set the acceptance multiplier k_L per image from how well the seed
    # stands out from its immediate surroundings (log-domain contrast). A tumor
    # that is clearly brighter than adjacent tissue can grow to a sharp edge with
    # a looser bound; a low-contrast tumor needs a tighter bound to avoid bleeding
    # into similar-intensity neighbours. k_G (drift ceiling) stays fixed.
    k_local = _k_local(L, mask, core, cfg, trace)

    for p in range(1, cfg.max_passes + 1):
        # Statistics frozen for the whole pass: updating per pixel lets each
        # borderline absorption widen sigma and admit the next one (leakage).
        mu_A = stats.mean
        s_A = stats.sigma()
        sigma_A = _sigma_A(s_A, sigma_floor, cfg.sigma_cap)
        t_local = k_local * sigma_A
        t_global = cfg.k_global * sigma_A
        n_requeued = n_rejected = 0

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
            ref = reference(r, c)
            d_now = abs(L[r, c] - ref)
            if d_now > d + cfg.lazy_tol:
                heapq.heappush(ssl, (d_now, r, c))
                n_requeued += 1
                continue

            # Global reference: the region mean frozen for the pass, or (cfg.global_running)
            # the running region mean. Healthy rule and bounds: _esrg_decide.
            ref_mu = stats.mean if cfg.global_running else mu_A
            act, tag = _esrg_decide(L[r, c], d_now, ref_mu, t_local, t_global, healthy_mu,
                                    cfg.healthy_rule, cfg.use_stopping)
            if record:
                events.append((p, int(r), int(c), float(L[r, c]), float(ref), float(d_now),
                               float(abs(L[r, c] - ref_mu)), tag))
            if act == "stop":
                # The SSL is sorted, so the first failure implies all the rest fail.
                stop = f"delta {d_now:.3f} > T_L {t_local:.3f}"
                break
            if act == "reject":
                # Global drift guard (or healthy veto): blocks chaining across a soft boundary.
                rejected[r, c] = True
                n_rejected += 1
                continue
            if act == "skip":
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
        if record:
            pass_detail.append({"pass": p, "n": stats.n - added, "mu": mu_A, "s": s_A,
                                "sigma_floor": sigma_floor, "sigma": sigma_A, "k_L": k_local,
                                "T_L": t_local, "T_G": t_global, "added": added,
                                "rejected": n_rejected, "requeued": n_requeued,
                                "area": int(region.sum()), "stop": stop})

        if added == 0:
            trace["stop_reason"] = f"converged after pass {p} (no pixels added)"
            break
    else:
        trace["stop_reason"] = f"reached P_max = {cfg.max_passes}"
    if trace["stop_reason"] is None:
        trace["stop_reason"] = f"converged after pass {p}"
    if record:
        trace["events"], trace["pass_detail"] = events, pass_detail

    return region, trace


def _esrg_decide(g, d, ref_mu, t_local, t_global, mu_H, rule, stopping):
    """
    ESRG's acceptance of one popped pixel, shared by ESRG and by SRG's region 1 (the
    SRG mirror of each applied method). Returns (action, tag):
      action  "absorb" | "reject" (never offered again) | "skip" (not this pass) |
              "stop" (local bound failed: in ESRG the sorted list ends the pass)
      tag     "absorb" | "rescue" | "veto" | "reject" | "skip" | "stop"   (for the GUI log)
    g        : pixel value;  d : its delta (LLDM reference, or region mean)
    ref_mu   : region mean the global test and the healthy rule compare with
    mu_H     : healthy prototype (None = rule unavailable);  rule : cfg.healthy_rule
    stopping : the adaptive stopping criterion is on
    With rule "off" this is exactly the earlier ESRG order: local bound, then global bound.
    """
    near = rule != "off" and mu_H is not None and abs(g - mu_H) < abs(g - ref_mu)
    if near and rule in ("veto", "both"):
        return "reject", "veto"               # nearer the healthy bulk than the region
    if not stopping:
        return "absorb", "absorb"
    if d > t_local:
        if rule in ("rescue", "both"):
            if abs(g - ref_mu) <= t_global and not near:
                return "absorb", "rescue"     # fails only the local bound and is not healthy-like
            return "skip", "skip"             # later entries may still be rescued: keep scanning
        return "stop", "stop"
    if abs(g - ref_mu) > t_global:
        return "reject", "reject"
    return "absorb", "absorb"


def _sigma_A(s_A, sigma_floor, sigma_cap=None):
    """sigma_A = max(s_A, floor); with a cap, min(that, max(cap, floor)) so the floor always wins."""
    sigma = max(s_A, sigma_floor)
    return sigma if not sigma_cap else min(sigma, max(sigma_cap, sigma_floor))


def _k_local(L, mask, core, cfg, trace):
    """k_L: fixed, or (cfg.adaptive_k) set from the seed's log-domain contrast with its ring."""
    if not cfg.adaptive_k:
        return cfg.k_local
    seed_mean = float(np.mean([L[r, c] for r, c in zip(*np.nonzero(core))]))
    ring = ndi.binary_dilation(core, morphology.disk(cfg.adaptive_ring)) & ~core & mask
    if ring.any():
        contrast = abs(seed_mean - float(L[ring].mean()))
        t = min(1.0, contrast / cfg.adaptive_ref_contrast)   # 0..1
        k_local = cfg.adaptive_k_lo + t * (cfg.adaptive_k_hi - cfg.adaptive_k_lo)
    else:
        k_local = cfg.k_local
    trace["adaptive_k_local"] = round(float(k_local), 3)
    return k_local


def _grow_esrg_compete(L, mask, core, sigma_floor, cfg, competitors, record=False, healthy_mu=None):
    """
    ESRG with competing seeds (cfg.esrg_compete). One SSL holds (delta, r, c, id):

      tumor (id 1)  ESRG rules exactly: delta against mu_loc (or mu_A with the local
                    measure off), lazy re-evaluation, 8-connected, mu_A / sigma_A
                    frozen per pass, absorption only within T_L and T_G when the
                    stopping criterion is on;
      competitor    the 1994 rule: delta = |L - running mean of the region|, fixed when
                    queued, unconditional absorption, connectivity of grow_srg
                    (cfg.srg_connectivity).

    The first tumor pixel with delta > T_L closes the tumor for the pass (the SSL is
    sorted), but the competitors keep growing, and a pixel beyond T_G is left to
    them. Competitors never stop, so they allocate every pixel they can reach (a
    declined pixel enclosed by the tumor stays unlabeled, as with SRG + ASC) and
    later passes add at most a few pixels.
    With the stopping criterion off this is a full tessellation in which only the
    tumor's measure and connectivity differ from SRG.
    """
    H, W = L.shape
    label = np.where(competitors >= 2, competitors, 0).astype(np.int32)
    label[core] = 1
    rejected = np.zeros((H, W), bool)
    trace = {"passes": [], "stop_reason": None, "compete": True}
    events, pass_detail = ([], []) if record else (None, None)

    local = _LocalMean(L.shape, cfg.local_radius)
    stats = _Welford()
    for r, c in zip(*np.nonzero(core)):
        local.add(r, c, L[r, c])
        stats.add(L[r, c])
    ids = [int(i) for i in np.unique(label) if i >= 2]
    sums = {i: float(L[label == i].sum()) for i in ids}
    counts = {i: int((label == i).sum()) for i in ids}
    if record:
        trace["seed_stats"] = {"n": stats.n, "mean": stats.mean, "s": stats.sigma()}

    def reference(r, c):
        if not cfg.use_log_local:
            return stats.mean
        m = local.mean(r, c)
        return stats.mean if m is None else m

    k_local = _k_local(L, mask, core, cfg, trace)
    comp_neigh = NEIGH8 if cfg.srg_connectivity == 8 else NEIGH4

    for p in range(1, cfg.max_passes + 1):
        mu_A, s_A = stats.mean, stats.sigma()
        sigma_A = _sigma_A(s_A, sigma_floor, cfg.sigma_cap)
        t_local, t_global = k_local * sigma_A, cfg.k_global * sigma_A
        n_requeued = n_rejected = 0
        ssl, tq = [], np.zeros((H, W), bool)       # tq: queued for the tumor this pass

        def push_tumor(r, c):
            for dr, dc in NEIGH8:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and mask[nr, nc] and label[nr, nc] == 0
                        and not rejected[nr, nc] and not tq[nr, nc]):
                    heapq.heappush(ssl, (abs(L[nr, nc] - reference(nr, nc)), nr, nc, 1))
                    tq[nr, nc] = True

        def push_comp(r, c, lid):
            ref = sums[lid] / counts[lid]
            for dr, dc in comp_neigh:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and label[nr, nc] == 0:
                    heapq.heappush(ssl, (abs(L[nr, nc] - ref), nr, nc, lid))

        for r, c in zip(*np.nonzero(label)):
            if label[r, c] == 1:
                push_tumor(r, c)
            else:
                push_comp(r, c, int(label[r, c]))

        added, tumor_open, stop = 0, True, "queue exhausted"
        while ssl:
            d, r, c, lid = heapq.heappop(ssl)
            if label[r, c] != 0:
                continue
            if lid != 1:                            # competitor: the 1994 rule
                label[r, c] = lid
                sums[lid] += float(L[r, c])
                counts[lid] += 1
                push_comp(r, c, lid)
                continue
            if not tumor_open or rejected[r, c]:
                continue
            ref = reference(r, c)
            d_now = abs(L[r, c] - ref)
            if d_now > d + cfg.lazy_tol:
                heapq.heappush(ssl, (d_now, r, c, 1))
                n_requeued += 1
                continue
            ref_mu = stats.mean if cfg.global_running else mu_A
            act, tag = _esrg_decide(L[r, c], d_now, ref_mu, t_local, t_global, healthy_mu,
                                    cfg.healthy_rule, cfg.use_stopping)
            if record:
                events.append((p, int(r), int(c), float(L[r, c]), float(ref), float(d_now),
                               float(abs(L[r, c] - ref_mu)), tag))
            if act == "stop":
                tumor_open = False                  # every later tumor entry fails too
                stop = f"delta {d_now:.3f} > T_L {t_local:.3f}"
                continue
            if act == "reject":
                rejected[r, c] = True               # left to the competitors
                n_rejected += 1
                continue
            if act == "skip":
                continue
            label[r, c] = 1
            local.add(r, c, L[r, c])
            stats.add(L[r, c])
            added += 1
            push_tumor(r, c)

        area = int((label == 1).sum())
        trace["passes"].append({"pass": p, "mu": round(mu_A, 4), "sigma": round(sigma_A, 4),
                                "T_L": round(t_local, 4), "T_G": round(t_global, 4),
                                "added": added, "area": area, "stop": stop})
        if record:
            pass_detail.append({"pass": p, "n": stats.n - added, "mu": mu_A, "s": s_A,
                                "sigma_floor": sigma_floor, "sigma": sigma_A, "k_L": k_local,
                                "T_L": t_local, "T_G": t_global, "added": added,
                                "rejected": n_rejected, "requeued": n_requeued,
                                "area": area, "stop": stop})
        if added == 0:
            trace["stop_reason"] = f"converged after pass {p} (no pixels added; competitors hold the rest)"
            break
    else:
        trace["stop_reason"] = f"reached P_max = {cfg.max_passes}"

    trace["labels"] = label
    trace["n_seed_regions"] = len(ids) + 1
    trace["unlabeled"] = int(((label == 0) & mask).sum())
    trace["region_areas"] = {1: int((label == 1).sum()), **{i: int(counts[i]) for i in ids}}
    trace["region_means"] = {1: round(stats.mean, 2), **{i: round(sums[i] / counts[i], 2) for i in ids}}
    if record:
        trace["events"], trace["pass_detail"] = events, pass_detail
    return label == 1, trace


class _Region1:
    """ESRG modules grafted onto SRG's tumor region (cfg.srg_use_log_local /
    srg_use_stopping). Only region 1 consults it; background regions keep the
    1994 rule, so they still claim whatever the tumor region declines.

    local_radius : None -> region mean (original); r -> mu_loc over the region-1
                   pixels in the (2r+1)^2 window around x (Eq. 3.8)
    stop         : None -> unconditional (original); dict(k_L, k_G, sigma_floor) ->
                   absorb only if delta <= k_L*sigma_A and |g - mu_A| <= k_G*sigma_A.
                   SRG has no passes, so sigma_A = max(s_A, sigma_floor) (optionally
                   capped, stop["sigma_cap"]) is read from
                   the running statistics at each decision; a declined pixel is never
                   offered to region 1 again.
    """

    def __init__(self, img, core, local_radius, stop, healthy=None):
        self.img, self.stop = img, stop
        self.healthy = healthy                 # dict(mu, rule) mirrors ESRG's healthy rule
        self.local = _LocalMean(img.shape, local_radius) if local_radius else None
        self.stats = _Welford()
        self.declined = np.zeros(img.shape, bool)
        for r, c in zip(*np.nonzero(core)):
            self.add(r, c)

    def add(self, r, c):
        if self.local is not None:
            self.local.add(r, c, self.img[r, c])
        self.stats.add(self.img[r, c])

    def reference(self, r, c, region_mean):
        if self.local is None:
            return region_mean
        m = self.local.mean(r, c)
        return region_mean if m is None else m

    def sigma(self):
        return (_sigma_A(self.stats.sigma(), self.stop["sigma_floor"], self.stop.get("sigma_cap"))
                if self.stop else None)

    def decide(self, r, c, d):
        """'absorb', or the reason region 1 declines x (which then stays declined).
        The same acceptance as ESRG (_esrg_decide) with SRG's running statistics."""
        if self.stop is None and self.healthy is None:
            return "absorb"
        sigma = self.sigma()
        t_l = self.stop["k_L"] * sigma if self.stop else None
        t_g = self.stop["k_G"] * sigma if self.stop else None
        h = self.healthy or {}
        act, tag = _esrg_decide(self.img[r, c], d, self.stats.mean, t_l, t_g, h.get("mu"),
                                h.get("rule", "off"), self.stop is not None)
        if act == "absorb":
            return "absorb"
        self.declined[r, c] = True
        return {"veto": "decline healthy", "reject": "decline >T_G"}.get(tag, "decline >T_L")

    def event(self, r, c, ref, d, verdict):
        """(row, col, g, reference, delta, |g - mu_A|, T_L, T_G, decision), GUI only."""
        g, sigma = float(self.img[r, c]), self.sigma()
        return (int(r), int(c), g, float(ref), float(d), abs(g - self.stats.mean),
                None if sigma is None else self.stop["k_L"] * sigma,
                None if sigma is None else self.stop["k_G"] * sigma, verdict)

    def to_trace(self, trace):
        if self.local is not None or self.stop is not None or self.healthy is not None:
            trace["declined"] = int(self.declined.sum())


def _seed_stats(img, core):
    n = int(core.sum())
    return {"n": n, "mean": float(img[core].mean()),
            "s": float(img[core].std(ddof=1)) if n > 1 else 0.0}


def _stop_reason(stop):
    return ("tessellation complete (no stopping criterion)" if stop is None
            else "tessellation complete (region 1 bounded by T_L and T_G)")


def grow_srg(img, seed_labels, record=False, local_radius=None, stop=None, connectivity=8,
             healthy=None):
    """
    Adams & Bischof (1994) seeded region growing, as published: 8-connected immediate
    neighbours (their Sec. II); connectivity=4 reproduces results computed before
    2026-10-13.

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

    local_radius / stop / healthy graft ESRG's applied methods onto region 1 only
    (_Region1), with ESRG's settings; all None (the default) is the published loop.

    Returns (label, trace). label is the full partition of img. Which id is
    "the structure of interest" is not decided here -- that is left to the
    caller, since SRG itself has no concept of a privileged region.
    """
    H, W = img.shape
    label = seed_labels.astype(np.int32).copy()

    ids = [int(i) for i in np.unique(label) if i > 0]
    sums = {i: float(img[label == i].sum()) for i in ids}
    counts = {i: int((label == i).sum()) for i in ids}
    r1 = _Region1(img, label == 1, local_radius, stop, healthy)
    neigh = NEIGH8 if connectivity == 8 else NEIGH4

    ssl = []
    # record (GUI only): the reference each region-1 entry was scored against.
    refs, events = ({}, []) if record else (None, None)

    def enqueue(r, c, lid):
        """delta is computed once, here, against the region mean of the moment."""
        if 0 <= r < H and 0 <= c < W and label[r, c] == 0:
            ref = sums[lid] / counts[lid]
            if lid == 1:
                if r1.declined[r, c]:
                    return
                ref = r1.reference(r, c, ref)
            d = abs(img[r, c] - ref)
            heapq.heappush(ssl, (d, r, c, lid))
            if record and lid == 1:
                refs[(d, r, c)] = ref

    for r, c in zip(*np.nonzero(label)):
        for dr, dc in neigh:
            enqueue(r + dr, c + dc, int(label[r, c]))

    while ssl:
        d, r, c, lid = heapq.heappop(ssl)
        if label[r, c] != 0:
            continue                    # a better-fitting region reached it first
        if lid == 1:
            if r1.declined[r, c]:
                continue
            verdict = r1.decide(r, c, d)
            if record:
                events.append(r1.event(r, c, refs[(d, r, c)], d, verdict))
            if verdict != "absorb":
                continue                # left to the other regions
            r1.add(r, c)
        label[r, c] = lid               # unconditional unless region 1 declined it
        sums[lid] += float(img[r, c])
        counts[lid] += 1
        for dr, dc in neigh:
            enqueue(r + dr, c + dc, lid)

    trace = {
        "stop_reason": _stop_reason(stop),
        "n_seed_regions": len(ids),
        "unlabeled": int((label == 0).sum()),
        "region_areas": {i: int(counts[i]) for i in ids},
        "region_means": {i: round(sums[i] / counts[i], 2) for i in ids},
    }
    r1.to_trace(trace)
    if record:
        trace["events"] = events
        trace["seed_areas"] = {i: int((seed_labels == i).sum()) for i in ids}
        if 1 in ids:
            trace["seed_stats"] = _seed_stats(img, seed_labels == 1)
    if len(ids) < 2 and stop is None:
        trace["warning"] = ("Only one seed region was planted. SRG has no stopping "
                            "rule, so with nothing to compete against, that single "
                            "region absorbs the entire image.")
    return label, trace
