"""
attribution.py — Per-module validation and failure attribution.

Purpose : Explain WHY an image scores as it does by testing each stage against
          ground truth and assigning one failure/success bucket.
Function : stage_metrics() computes a GT-referenced metric per stage; attribute()
          walks the stages in order and returns the first failing gate (A head mask,
          B candidates, C seed, D leak, E undersegment, F post-proc, G success).
Notes   : Bucket thresholds are fixed and documented so attribution is deterministic;
          this is the instrument that showed seeding + thresholding dominate failures.
"""
import numpy as np

# Thresholds that define the buckets. Documented and fixed so attribution is deterministic.
SUCCESS_DSC = 0.70      # bucket G
SEED_HIT_FRAC = 0.50    # >= this share of seed core inside GT counts as a "hit" for attribution
CAND_COVERAGE = 0.30    # tumor must have >= this share of its pixels in the candidate set
LEAK_PRECISION = 0.50   # below this on a seed-correct image = leakage (bucket D)
UNDER_RECALL = 0.50     # below this on a seed-correct image = undersegmentation (bucket E)
POSTPROC_DROP = 0.10    # DSC drop caused by post-processing that flags bucket F


def _cover(pred, gt):
    """Share of GT pixels covered by pred (recall of pred w.r.t. gt)."""
    gt = gt.astype(bool)
    g = int(gt.sum())
    return float((pred.astype(bool) & gt).sum()) / g if g else np.nan


def _dsc(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 2.0 * (pred & gt).sum() / denom if denom else np.nan


def stage_metrics(result):
    """Ground-truth-referenced metric for each stage. Returns a flat dict."""
    gt = result.gt
    if gt is None:
        return {}
    S = result.stage
    m = {}

    # Stage 2: does the head mask retain the tumor?
    head = S("mask").image.astype(bool)
    m["s2_tumor_retention"] = _cover(head, gt)
    m["s2_headmask_frac"] = float(head.mean())

    # Stage 4: is the tumor present in the retained candidate class?
    cand = S("candidates").image.astype(bool)
    m["s4_tumor_in_cand"] = _cover(cand, gt)

    # Stage 5: seed placement
    seed = S("seed").image.astype(bool)
    m["s5_seed_area"] = int(seed.sum())
    m["s5_seed_in_gt_frac"] = (float((seed & gt).sum()) / seed.sum()
                               if seed.sum() else np.nan)
    m["s5_seed_hit"] = (bool((seed & ~gt.astype(bool)).sum() == 0)
                        if seed.sum() else None)

    # Stage 6: growth quality (meaningful only if seed is correct; computed always, filtered later)
    growth = S("growth").image.astype(bool) if S("growth") else None
    if growth is not None:
        tp = (growth & gt).sum()
        m["s6_precision"] = float(tp) / growth.sum() if growth.sum() else np.nan
        m["s6_recall"] = float(tp) / gt.sum() if gt.sum() else np.nan
        m["s6_dsc"] = _dsc(growth, gt)

    # Stage 7: did post-processing help or hurt?
    final = result.mask.astype(bool)
    m["s7_dsc"] = _dsc(final, gt)
    if growth is not None:
        m["s7_dsc_delta"] = m["s7_dsc"] - m["s6_dsc"]

    return m


def attribute(result):
    """
    Walk the stages in order and return (bucket, reason).
    First failing gate wins, so each image lands in exactly one bucket.
    """
    gt = result.gt
    if gt is None or gt.sum() == 0:
        return "?", "no ground truth"

    m = stage_metrics(result)

    # System explicitly reported no tumor, but GT has one -> a miss, blame candidate/seed stage
    if result.status == "NO TUMOR CANDIDATE":
        return "X", "system flagged no-candidate but tumor is present"

    # A: tumor destroyed by head masking
    if m["s2_tumor_retention"] < 0.90:
        return "A", f"head mask kept only {m['s2_tumor_retention']:.0%} of tumor"

    # B: tumor never entered the candidate set (thresholding problem)
    if m["s4_tumor_in_cand"] < CAND_COVERAGE:
        return "B", f"tumor coverage in candidates {m['s4_tumor_in_cand']:.0%} < {CAND_COVERAGE:.0%}"

    # C: a usable candidate existed but the seed landed off-tumor (ranking problem)
    seed_ok = (m["s5_seed_in_gt_frac"] is not np.nan
               and m["s5_seed_in_gt_frac"] >= SEED_HIT_FRAC)
    if not seed_ok:
        return "C", (f"candidate coverage was {m['s4_tumor_in_cand']:.0%} but seed only "
                     f"{(m['s5_seed_in_gt_frac'] if m['s5_seed_in_gt_frac']==m['s5_seed_in_gt_frac'] else 0):.0%} inside GT")

    # Seed is correct from here on. Judge growth and post-processing.
    final_dsc = m["s7_dsc"]
    if final_dsc >= SUCCESS_DSC:
        return "G", f"success, DSC {final_dsc:.2f}"

    # F: post-processing measurably hurt an otherwise-good growth
    if "s7_dsc_delta" in m and m["s7_dsc_delta"] <= -POSTPROC_DROP:
        return "F", f"post-processing dropped DSC by {-m['s7_dsc_delta']:.2f}"

    # D vs E: leakage vs undersegmentation, decided on the growth output
    prec = m.get("s6_precision", np.nan)
    rec = m.get("s6_recall", np.nan)
    if prec < LEAK_PRECISION and (np.isnan(rec) or prec <= rec):
        return "D", f"leakage: growth precision {prec:.2f}"
    if rec < UNDER_RECALL:
        return "E", f"undersegmentation: growth recall {rec:.2f}"
    # Seed correct, growth reasonable, but final DSC still < success threshold
    return "E", f"partial: DSC {final_dsc:.2f}, precision {prec:.2f}, recall {rec:.2f}"


BUCKET_LABEL = {
    "A": "Head mask lost tumor (Stage 2)",
    "B": "Tumor absent from candidates (Stage 4)",
    "C": "Seed chose wrong structure (Stage 5)",
    "D": "Growth leaked (Stage 6 stopping)",
    "E": "Growth undersegmented (Stage 6 measure)",
    "F": "Post-processing hurt result (Stage 7)",
    "G": "Success",
    "X": "False no-candidate flag",
    "?": "No ground truth",
}
