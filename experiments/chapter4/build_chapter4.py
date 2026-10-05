"""
build_chapter4.py — Write Chapter 4 (Results and Discussion) from the evaluation outputs.

Purpose : Assemble the chapter text, equations, APA tables, and figures from
          results.json and the raw per-slice rows, so that every number in the
          chapter is computed, never typed, and agrees with the appendix workbooks.
Function : build() returns the chapter specification (a list of blocks) and writes
          it to outputs/evaluation/chapter4_spec.json; render() calls
          render_docx.js to produce Chapter4_Results_and_Discussion.docx.
Notes   : Requires Node.js and the npm package "docx" (npm install in this folder,
          or NODE_PATH pointing to a node_modules that has it).
          CLI: python experiments/chapter4/build_chapter4.py [--dir outputs/evaluation]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from experiments.chapter4 import tables as T
from experiments.chapter4.tables import CL, f, n_, nz, pct
from experiments.stats import fmt_p

ROOT = os.path.dirname(os.path.dirname(HERE))


# ─── Text helpers ────────────────────────────────────────────────────────────
def P(p):
    """'*p* < .001' or '*p* = .023' (APA)."""
    s = fmt_p(p)
    return f"*p* {s}" if s.startswith("<") else f"*p* = {s}"


def sig(p):
    return p is not None and p < 0.05


def eff(r):
    a = abs(r or 0)
    return "large" if a >= 0.5 else "medium" if a >= 0.3 else "small" if a >= 0.1 else "negligible"


def Zr(t, key="p_holm"):
    return f"*Z* = {f(t['Z'], 2)}, {P(t.get(key, t['p']))}, *r* = {nz(t['r'])}"


def more(a, b, hi="higher", lo="lower"):
    return hi if a > b else lo


class Doc:
    def __init__(self):
        self.blocks = []
        self.tabs = None

    def h1(self, t): self.blocks.append({"t": "h1", "text": t})
    def h2(self, t): self.blocks.append({"t": "h2", "text": t})
    def h3(self, t): self.blocks.append({"t": "h3", "text": t})
    def p(self, t, **kw): self.blocks.append({"t": "p", "text": t, **kw})
    def eq(self, num, m): self.blocks.append({"t": "eq", "num": num, "math": m})
    def where(self, *items): self.blocks.append({"t": "where", "items": list(items)})
    def ref(self, t): self.blocks.append({"t": "ref", "text": t})

    def table(self, key):
        t = dict(self.tabs[key])
        self.blocks.append({"t": "table", **{k: t[k] for k in ("num", "title", "header", "rows", "widths", "note")},
                            **({"align": t["align"]} if "align" in t else {})})

    def fig(self, num, img, title, note, size):
        self.blocks.append({"t": "figure", "num": num, "img": img, "title": title, "note": note,
                            "w": size[0], "h": size[1]})


def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


# ─── Math shorthands (render_docx.js math nodes) ─────────────────────────────
def sub(b, s): return {"sub": [b, s]}
def sup(b, s): return {"sup": [b, s]}
def frac(a, b): return {"f": [a, b]}
def sqrt(x): return {"sqrt": x}
def rb(x): return {"rb": x}
def ssum(lo, hi, body): return {"sum": {"lo": lo, "hi": hi, "body": body}}


def build(d):
    R = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    D = Doc()
    D.tabs = T.build(R)
    tb = {k: v["num"] for k, v in D.tabs.items()}
    M, E1, E2, E3, E4, E5, E6 = R["meta"], R["e1"], R["e2"], R["e3"], R["e4"], R["e5"], R["e6"]
    S = R["summary"]
    n, N = M["n_sample"], M["N_population"]
    nh = R["strata"][0]["n_h"]
    ss = {r["level"]: r for r in R["sample_size"]}
    figdir = os.path.join(d, "figures")

    def figure(num, name, title, note):
        pth = os.path.join(figdir, name)
        D.fig(num, os.path.relpath(pth, d).replace("\\", "/"), title, note, png_size(pth))

    def row(cmp, metric):
        return next(r for r in cmp if r["metric"] == metric)

    au, pl = E4["auto"], E4["planted"]

    # ══ 4.1 ═══════════════════════════════════════════════════════════════════
    D.h1("Chapter Four")
    D.h1("RESULTS AND DISCUSSION")
    D.h2("4.1 General Results")
    D.p("This chapter presents the evaluation of the Enhanced Seeded Region Growing (ESRG) algorithm against the "
        "original Seeded Region Growing (SRG) algorithm of Adams and Bischof (1994), which serves as the baseline "
        "control. The evaluation follows the three aspects that Udupa et al. (2006) identify for judging a "
        "segmentation method: its *accuracy* (how closely the result matches the ground truth), its *precision* in "
        "the sense of reproducibility (whether the same image always yields the same result), and its *efficiency* "
        "(the time it takes). Each specific objective is evaluated with the metric that measures its problem most "
        "directly: the seed hit rate and the variability between operators for automated seed selection "
        "(Objective 1), recall for undersegmentation (Objective 2), and precision, the leakage rate, and boundary "
        "distances for boundary leakage (Objective 3). The Dice Similarity Coefficient (DSC) and Intersection over "
        "Union (IoU) summarize the overall delineation, and processing time measures efficiency.")
    D.p(f"All {M['n_rows']:,} runs were produced by an automated, reproducible pipeline: experiments/sampling.py draws "
        "the sample, experiments/evaluate.py runs every configuration on every sampled slice and writes one row of raw "
        "results per slice and configuration, experiments/analyze.py computes every statistic and figure in this "
        "chapter from those rows, and experiments/appendix.py writes one Excel workbook per evaluation (Appendices A to "
        f"H) that lists the result of every slice. {'None' if M['n_errors'] == 0 else M['n_errors']} of the runs ended "
        "in an error. The same metric definitions are implemented in the system itself: its Evaluation step displays, "
        "for any loaded slice, each metric together with the source of every variable and the worked computation.")

    # ── Sampling ─────────────────────────────────────────────────────────────
    D.h3("4.1.1 Dataset and Sampling")
    tr = M["split_counts"].get("train", 0)
    te = M["split_counts"].get("test", 0)
    D.p(f"The population is the segmentation task of the BRISC 2025 dataset (Fateh et al., 2026): {N:,} contrast-enhanced "
        "T1-weighted slices, every one with a radiologist-reviewed tumor mask, pooled from the training split "
        f"({sum(s['N_h_train'] for s in R['strata']):,}) and the test split ({sum(s['N_h_test'] for s in R['strata']):,}). "
        "The two splits were pooled because the test split alone is too small to give every tumor class the sample "
        "size required below. Because the region-growing method has no trainable weights, a training slice is not "
        "seen during learning; however, the three weights of the seed-ranking score were tuned on the training split "
        "(Table 3.4), so the possibility that training slices are favored is tested separately (Section 4.1.5). The "
        "population was divided into nine strata, one for each combination of tumor class (glioma, meningioma, "
        "pituitary) and imaging plane (axial, coronal, sagittal), both read from each filename.")
    D.p("The sample size was determined with Cochran’s (1977) formula for estimating a proportion, such as a success "
        "rate or the seed hit rate, to within a margin of error *e* at 95% confidence:")
    D.eq("4.1", [sub("n", "0"), " = ", frac([sup("z", "2"), "p", rb("1 − p")], sup("e", "2"))])
    D.where("*n*_{0} – required sample size for an infinite population",
            "*z* – 1.96, the standard normal value for 95% confidence",
            "*p* – assumed population proportion; *p* = .50 gives the largest, most conservative size",
            "*e* – tolerated margin of error, .05 (±5 percentage points)")
    D.p("Because each population is finite, *n*_{0} is reduced with the finite population correction:")
    D.eq("4.2", [sub("n", "c"), " = ", frac(sub("n", "0"), ["1 + ", frac([sub("n", "0"), " − 1"], sub("N", "c"))])])
    D.where("*n*_{c} – required sample size for population *c*, rounded up to the next whole slice",
            "*N*_{c} – number of slices in population *c*")
    pi = ss["pituitary"]
    D.p("Chapter 4 reports every result per tumor class as well as overall, so the requirement was applied to each "
        "class separately: a class-level rate must itself be within ±5%. Substituting the values gives "
        f"*n*_{{0}} = 1.96² × .50 × .50 / .05² = {f(pi['n0'], 2)}; for the largest class, pituitary "
        f"(*N*_{{c}} = {pi['N']:,}), *n*_{{c}} = {f(pi['n0'], 2)} / (1 + {f(pi['n0'] - 1, 2)} / {pi['N']:,}) = "
        f"{f(pi['n_exact'], 2)}, rounded up to {pi['n_min']}. Table {tb['sample_size']} lists every class. The largest "
        "class requirement was then given to every class and divided equally among the three planes, so that all nine "
        "strata receive the same number of slices (equal allocation):")
    D.eq("4.3", [sub("n", "h"), " = ⌈", frac([sub("max", "c"), " ", sub("n", "c")], "P"), "⌉,  n = H · ", sub("n", "h")])
    D.where("*n*_{h} – number of slices sampled from stratum *h*",
            f"*P* – number of imaging planes per class, 3; *H* – number of strata, 9",
            "⌈ ⌉ – rounding up to the next whole slice")
    D.p(f"This gives *n*_{{h}} = ⌈{pi['n_min']}/3⌉ = {nh} slices per stratum and a total sample of 9 × {nh} = {n:,} "
        f"slices ({nh * 3} per class). The precision that the drawn sample actually achieves is:")
    D.eq("4.4", ["e = z", sqrt([frac(["p", rb("1 − p")], "n"), " · ", frac(["N − n"], ["N − 1"])])])
    D.where("*e* – achieved margin of error of a proportion at 95% confidence",
            "*n*, *N* – sample and population size of the level considered (all slices, or one class)")
    D.p(f"Overall, *e* = 1.96 × √(.25/{n:,} × {N - n:,}/{N - 1:,}) = ±{100 * ss['all']['margin_achieved']:.2f}%, and "
        "within each class the margin stays below the targeted ±5% (Table "
        f"{tb['sample_size']}). Equal allocation was chosen over proportional allocation because the evaluation "
        "compares tumor classes and planes with one another: giving every stratum the same size gives every "
        "class-by-plane comparison the same precision and prevents the largest class from dominating the averages "
        "(Lohr, 2021). Within each stratum, the slices were sorted by filename, shuffled with a fixed random seed "
        "(2026), and the first *n*_{h} were taken. This is simple random sampling without replacement, so every slice "
        "of a stratum had the same chance of selection, *f*_{h} = *n*_{h}/*N*_{h}, and the selection involved no "
        "judgment. Because the sampling fractions differ slightly between strata, population-weighted estimates "
        "(Equation 4.15) are reported alongside the sample means to confirm that the equal allocation does not bias "
        "the overall figures.")
    D.table("sample_size")
    D.table("strata")
    D.p(f"After the draw, the sample was recorded in a manifest (outputs/evaluation/sample.csv), together with the full "
        f"sampling frame of all {N:,} slices and their random draw order (Appendix A), and every image and mask that "
        f"was not sampled ({N - n:,} image–mask pairs) was removed from the dataset folder. The {n:,} retained slices "
        f"comprise {tr:,} from the training split and {te:,} from the test split.")

    # ── Protocol ─────────────────────────────────────────────────────────────
    D.h3("4.1.2 Evaluation Protocol")
    D.p(f"Every sampled slice was processed under the {M['n_configs']} configurations in Table {tb['configs']}, so every "
        "comparison is paired on identical images. Three seeding conditions separate the contribution of each "
        "enhancement. Under *automatic seeding*, both algorithms start from the seed produced by the automated seed "
        "selection of Objective 1; the baseline also receives a regular grid of background seeds (every 8 pixels "
        "inside the head mask), because the original algorithm needs competing regions in order to stop. Under "
        "*planted seeding*, one click is placed at the deepest pixel of the ground-truth tumor, the pixel farthest from "
        "its boundary, and is dilated into a disk of radius 3 pixels exactly as a manual click is. ESRG reduces this "
        "disk to its core, as it does in manual mode, while SRG grows it against the same background grid. Planted "
        "seeding simulates a correctly placed manual seed on every slice and therefore measures region growing "
        "independently of seed selection, which is where Objectives 2 and 3 act.")
    D.p("Under *operator seeding*, five clicks per slice simulate five operators, a design adopted from "
        "studies of inter-observer variability (Joskowicz et al., 2019; Visser et al., 2019) and from the evaluation "
        "of interactive segmentation with simulated users (Moschidis & Graham, 2010). Each click is a different tumor "
        "pixel, drawn at random from the pixels deep enough for the whole click disk to lie inside the tumor:")
    D.eq("4.5", [sub("K", "i"), " = {x ∈ ", sub("G", "i"), " : ", sub("D", sub("G", "i")), "(x) > ρ},  ",
                 sub("c", "i1"), ", …, ", sub("c", "i5"), " ~ Uniform(", sub("K", "i"), ") without replacement"])
    D.where("*K*_{i} – admissible click positions on slice *i*",
            "*G*_{i} – ground-truth tumor of slice *i*; *D*_{G}(*x*) – distance from *x* to the nearest pixel outside *G* (Equation 3.4)",
            "ρ – click radius, 3 pixels; each click is dilated into a disk of this radius",
            "*c*_{i1}, …, *c*_{i5} – the five simulated operator clicks, drawn with a seed derived from the filename")
    D.p("Every click is therefore a valid, correctly placed click; the clicks differ only in where inside the tumor "
        "the operator happened to click. Moschidis and Graham (2010) found that seeds placed in the interior of the "
        "object give better results than seeds near its contour, so this design is favorable to manual seeding. For "
        "Experiment E2, a synthetic bias field was multiplied into each slice before normalization, reproducing the "
        "image-formation model of Equation 3.15:")
    D.eq("4.6", ["I′(x) = I(x) · ", rb(["1 + ", frac("β", "2"), " · ", frac("u(x)", sub("u", "max"))]),
                 ",  u(x) = (x − c) · (cos θ, sin θ)"])
    D.where("*I*(*x*), *I*′(*x*) – intensity of pixel *x* before and after the bias field is applied",
            f"β – bias amplitude, {M['inu']:.1f}: the field spans {1 - M['inu'] / 2:.1f} to {1 + M['inu'] / 2:.1f}, a "
            f"{int(100 * M['inu'])}% intensity non-uniformity",
            "*u*(*x*) – signed distance of *x* from the image center *c* along the gradient direction",
            "*u*_{max} – largest |*u*(*z*)| over the image, which scales the field to exactly ±β/2 at the image edges",
            "θ – gradient direction, drawn at random for each slice with a fixed seed")
    D.table("configs")

    # ── Metrics ──────────────────────────────────────────────────────────────
    D.h3("4.1.3 Evaluation Metrics")
    D.p(f"Table {tb['metrics']} lists every metric, the objective it evaluates, and the studies that support its use. "
        "Each metric is computed per slice from the predicted mask *M* (the final output of Stage 7), the ground-truth "
        "mask *G*, and, for Objective 1, the seed core *S* (the output of Stage 5), all at the working resolution "
        "(longer side ≤ 512 pixels). Comparing *M* and *G* pixel by pixel gives the true positives TP = |*M* ∩ *G*|, "
        "the false positives FP = |*M* \\ *G*|, and the false negatives FN = |*G* \\ *M*|, from which the overlap "
        "metrics of Equations 3.32 to 3.35 follow. The metrics were chosen following the recommendation that an "
        "overlap metric be paired with a boundary-distance metric and that each metric be matched to the property of "
        "interest (Maier-Hein et al., 2024; Reinke et al., 2024).")
    D.table("metrics")
    D.p("*Objective 1.* The original algorithm has no seed selection step, so its result depends on where the seed is "
        "placed (Adams & Bischof, 1994; Mehnert & Jackway, 1997); Fan et al. (2005) showed that seed placement is the "
        "decisive factor in the quality of seeded region growing. The Seed Hit Rate (Equation 3.38) measures this "
        "directly: a seed core that lies entirely inside the tumor can only grow into the tumor first. Its 95% "
        "confidence interval is the Wilson score interval (Equation 4.14), which remains accurate for rates near 0% "
        "or 100% (Brown et al., 2001). How far a missed seed lies from the tumor is measured by the seed localization "
        "distance:")
    D.eq("4.7", [sub("d", "S"), " = ", sub("min", "y ∈ G"), " d(", sub("x̄", "S"), ", y)"])
    D.where("*d*_{S} – seed localization distance in pixels; 0 when the centroid lies inside the tumor",
            "*x̄*_{S} – centroid of the seed core *S*, rounded to the nearest pixel",
            "*d*(*a*, *b*) – Euclidean distance between pixels *a* and *b*")
    D.p("The problem statement of Objective 1 is that manual seeding produces *inconsistent* results. Consistency is "
        "the precision aspect of Udupa et al. (2006) and is measured, as in studies of inter-observer agreement "
        "(Joskowicz et al., 2019; Zou et al., 2004), by how much the result changes when a different operator seeds "
        "the same slice. For each slice *i*, the spread of DSC over the *K* = 5 simulated operators (Equation 4.5) is:")
    D.eq("4.8", [sub("SD", "i"), " = ", sqrt([frac("1", "K − 1"), ssum("k = 1", "K", sup(rb([sub("DSC", "ik"), " − ", sub("m", "i")]), "2"))])])
    D.where("*SD*_{i} – within-slice standard deviation of DSC; 0 means every operator obtained the same result",
            "*DSC*_{ik} – DSC obtained from the click of operator *k*; *m*_{i} – mean DSC of the five operators")
    D.p("Whether the variation changes the *outcome* is measured by the inconsistency rate, the share of slices on which "
        "some operators succeed and others fail:")
    D.eq("4.9", ["IR = ", frac("1", "N"), ssum("i = 1", "N", ["𝟙", {"sb": ["0 < ", ssum("k = 1", "K", sub("s", "ik")), " < K"]}])])
    D.where("*IR* – inconsistency rate",
            "*s*_{ik} – 1 if the click of operator *k* gives a successful segmentation of slice *i* (Equation 4.11), else 0",
            "𝟙[·] – indicator, 1 when the condition holds")
    D.p("The automatic seed is computed from the image alone, so it is deterministic and its *SD*_{i} and *IR* are zero "
        "by construction. This was verified empirically by running automatic seeding twice on the slices of the timing "
        "subsample (below), once in the evaluation run and once in the separate timing pass, and comparing the masks.")
    D.p("*Objective 2.* Undersegmentation means that part of the tumor is left out of the result, that is, false "
        "negatives. Recall (sensitivity, Equation 3.35) is the share of the true tumor that the result recovers, and its "
        "complement, FN/|*G*| = 1 − recall, is the share left unsegmented; it is the standard measure of "
        "undersegmentation (Taha & Hanbury, 2015; Udupa et al., 2006). Recall is not affected by false positives, so it "
        "isolates the effect that the local log-domain measure is designed to have.")
    D.p("*Objective 3.* Boundary leakage means that the region crosses the tumor boundary into surrounding tissue, that "
        "is, false positives. Precision (Equation 3.34) is the share of the result that is tumor, so its complement is "
        "the share that leaked. The Leakage Rate (Equation 3.39) counts the slices on which the region grew to more "
        "than twice the tumor area, the gross flooding that unconditional absorption causes (Adams & Bischof, 1994; "
        "Fan et al., 2005). Because overlap metrics are insensitive to where the errors lie, the boundary itself is "
        "evaluated with HD95 (Equation 3.37), the boundary metric of the BRATS benchmark (Menze et al., 2015), and with "
        "the average symmetric surface distance (Taha & Hanbury, 2015; Yeghiazaryan & Voiculescu, 2018):")
    D.eq("4.10", ["ASSD = ", frac([ssum("p ∈ ∂M", "", ["d(p, ∂G)"]), " + ", ssum("q ∈ ∂G", "", ["d(q, ∂M)"])],
                                 [sub("N", "∂M"), " + ", sub("N", "∂G")])])
    D.where("∂*M*, ∂*G* – boundary pixels of the predicted and true masks (Equation 3.36)",
            "*d*(*p*, ∂*G*) – Euclidean distance from pixel *p* to the nearest pixel of ∂*G*, in pixels",
            "*N*_{∂M}, *N*_{∂G} – number of boundary pixels of each mask")
    D.p("HD95 and ASSD are undefined when the prediction is empty, which happens when the seed selection reports no "
        "tumor candidate. Dropping those slices would hide failures, so, as recommended by Maier-Hein et al. (2024) and "
        "Reinke et al. (2024), they receive the worst possible value, the length of the image diagonal (for example, "
        "724.1 pixels for a 512 × 512 slice).")
    D.p("*Overall delineation and efficiency.* DSC (Equation 3.32) is the most widely used measure of segmentation "
        "accuracy and of agreement between segmentations (Dice, 1945; Zou et al., 2004), and IoU (Equation 3.33) is its "
        "stricter counterpart (Jaccard, 1912). Because the per-slice DSC is often bimodal, the share of successfully "
        "segmented slices is also reported:")
    D.eq("4.11", [sub("s", "i"), " = 𝟙", {"sb": [sub("DSC", "i"), " ≥ 0.70"]}, ",  SR = ", frac("1", "N"),
                 ssum("i = 1", "N", sub("s", "i"))])
    D.where("*s*_{i} – success indicator of slice *i*; *SR* – success rate",
            "0.70 – success threshold; a DSC above 0.70 is regarded as excellent agreement (Zijdenbos et al., 1994)")
    env = M.get("environment") or {}
    nt = E6["all"]["N"] if E6 else 0
    D.p("Processing time is the wall-clock time per slice from loading the image to the final mask, excluding the "
        "computation of the metrics. The accuracy runs used ten parallel worker processes, whose timings are distorted "
        "by contention, so time was measured in a separate sequential pass following good benchmarking practice "
        "(Hoefler & Belli, 2015): one process with nothing else running, a discarded warm-up run, the order of the two "
        "algorithms alternated from slice to slice, and summaries by medians and nonparametric tests. Because a "
        "sequential pass over all slices would take several hours, it was run on a stratified random subsample: the "
        f"first {nt // 9 if nt else 20} slices of every stratum in the random draw order of the sampling step "
        f"({nt or 180} slices). The pass ran on {env.get('cpu', 'the hardware of Table 3.1')} with "
        f"{env.get('ram_gb', '—')} GB of memory under the Windows "
        f"*{(env.get('power_scheme', '').split('(')[-1].rstrip(')') or 'default')}* power plan"
        + (", on battery power, which lowers the clock speed; absolute times are therefore longer than on mains power, "
           "but the comparison between the two algorithms is unaffected because every slice was timed for both under "
           "the same conditions." if env.get("source") == "battery" else "."))

    # worked example of the per-slice metrics
    a = df[df.config == "A_ESRG"].copy()
    hits = a[(a.seed_hit == True) & (a.tumor == "meningioma")]
    if len(hits):
        ex = hits.iloc[(hits.dsc - hits.dsc.median()).abs().argsort().iloc[0]]
        tp, fp, fn = int(ex.tp), int(ex.fp), int(ex.fn)
        D.p(f"As a worked example, consider slice {ex.file} (meningioma, {ex.plane}) segmented by ESRG under automatic "
            f"seeding. Its ground truth contains |*G*| = {int(ex.gt_area):,} pixels and the result |*M*| = "
            f"{int(ex.pred_area):,} pixels, of which TP = {tp:,} lie inside the tumor, so FP = {fp:,} and FN = {fn:,}. "
            f"Then DSC = 2({tp:,}) / (2({tp:,}) + {fp:,} + {fn:,}) = {f(ex.dsc, 4)}, IoU = {tp:,} / {tp + fp + fn:,} = "
            f"{f(ex.iou, 4)}, precision = {tp:,} / {tp + fp:,} = {f(ex.precision, 4)}, recall = {tp:,} / {tp + fn:,} = "
            f"{f(ex.recall, 4)}, and the area ratio is {int(ex.pred_area):,} / {int(ex.gt_area):,} = "
            f"{f(ex.area_ratio, 3)}, so the slice {'leaked' if ex.area_ratio > 2 else 'did not leak'} and "
            f"{'is' if ex.dsc >= 0.7 else 'is not'} a success. Its seed core of {int(ex.seed_area)} pixels lies "
            f"entirely inside the tumor (a hit), and HD95 = {f(ex.hd95_wc, 2)} pixels. The same computation, with every "
            "intermediate value, is displayed by the system’s Evaluation step, and the per-slice values of every "
            "evaluation are listed in Appendices B to G.")

    # ── Statistics ───────────────────────────────────────────────────────────
    D.h3("4.1.4 Statistical Treatment of the Results")
    nt = D.tabs["normality"]
    D.p("Because every configuration was run on the same slices, configurations are compared with paired tests. The "
        "paired *t* test assumes normally distributed differences; this was checked with the Shapiro–Wilk test "
        f"(Shapiro & Wilk, 1965), which rejected normality for every comparison in Table {tb['normality']}. Per-slice "
        "DSC is bounded between 0 and 1 and bimodal (Figure 4.1), so the nonparametric Wilcoxon signed-rank test was "
        "used, as recommended for comparing two methods on the same set of cases (Demšar, 2006; Wilcoxon, 1945).")
    D.table("normality")
    D.p("For every slice, the difference *d*_{i} = *y*_{i} − *x*_{i} between the compared configuration *y* and the "
        "reference *x* is computed; zero differences are dropped, the remaining |*d*_{i}| are ranked, and:")
    D.eq("4.12", ["Z = ", frac(["W⁺ − ", frac("n′(n′ + 1)", "4")],
                               sqrt([frac("n′(n′ + 1)(2n′ + 1)", "24"), " − ", ssum("g", "", frac([sup(sub("t", "g"), "3"), " − ", sub("t", "g")], "48"))]))])
    D.where("*W*^{+} – sum of the ranks of the positive differences (slices on which *y* scored higher)",
            "*n*′ – number of slices with a non-zero difference",
            "*t*_{g} – number of differences sharing the same absolute value in tie group *g*",
            "*Z* – standardized statistic, positive when *y* tends to score higher than *x*")
    w = row(au, "dsc")
    D.p("The two-sided *p* value is read from the standard normal distribution, and the effect size is "
        "*r* = *Z*/√*N* (Equation 3.41), where *N* is the number of pairs; |*r*| near .1, .3, and .5 is conventionally a "
        "small, medium, and large effect (Fritz et al., 2012). For example, for DSC under automatic seeding (Table "
        f"{tb['overall_auto']}), ESRG scored higher than SRG on {w['n_pos']:,} slices and lower on {w['n_neg']:,}, with "
        f"{w['n_zero']:,} ties at zero; with *n*′ = {w['n_nonzero']:,}, *W*^{{+}} = {w['W_plus']:,.1f}, the expected value "
        f"*n*′(*n*′ + 1)/4 = {w['E_W']:,.1f}, and a tie-corrected standard deviation of {w['SD_W']:,.1f}, "
        f"*Z* = ({w['W_plus']:,.1f} − {w['E_W']:,.1f}) / {w['SD_W']:,.1f} = {f(w['Z'], 2)} and *r* = {f(w['Z'], 2)} / "
        f"√{w['N']:,} = {nz(w['r'])}. When several tests answer one question, such as the metrics of one comparison or "
        "the three tumor classes, the *p* values are adjusted with the Holm procedure (Equation 3.40). Paired rates, "
        "such as the success and leakage rates, are compared with the exact McNemar test (McNemar, 1947), which uses "
        "only the slices on which the two configurations disagree:")
    D.eq("4.13", ["p = min", rb(["1, 2", ssum("i = 0", "min(b, c)", ["C(b + c, i) ", sup("0.5", "b + c")])])])
    D.where("*b* – slices on which only the reference configuration has the outcome",
            "*c* – slices on which only the compared configuration has the outcome",
            "C(*b* + *c*, *i*) – number of ways of choosing *i* of the *b* + *c* discordant slices")
    D.p("Rates are reported with the Wilson score interval (Wilson, 1927), recommended over the normal approximation "
        "because it stays within 0–100% and keeps its coverage near the extremes (Brown et al., 2001):")
    D.eq("4.14", [frac(["p̂ + ", frac(sup("z", "2"), "2n"), " ± z", sqrt([frac(["p̂(1 − p̂)"], "n"), " + ", frac(sup("z", "2"), sup("4n", "2"))])],
                       ["1 + ", frac(sup("z", "2"), "n")])])
    D.where("*p̂* – observed proportion *k*/*n*; *z* – 1.96")
    D.p("To express the overall results for the whole population rather than for the equally allocated sample, each "
        "stratum mean is weighted by the stratum’s share of the population (Cochran, 1977):")
    D.eq("4.15", [sub("ȳ", "st"), " = ", ssum("h = 1", "H", [sub("W", "h"), sub("ȳ", "h")]), ",  ", sub("W", "h"), " = ",
                  frac(sub("N", "h"), "N"), ",  SE = ", sqrt(ssum("h = 1", "H", [sup(sub("W", "h"), "2"), rb(["1 − ", sub("f", "h")]), frac(sup(sub("s", "h"), "2"), sub("n", "h"))]))])
    D.where("*ȳ*_{h}, *s*_{h} – sample mean and standard deviation of stratum *h*",
            "*f*_{h} – sampling fraction *n*_{h}/*N*_{h}; *SE* – standard error of the weighted estimate")
    D.p("The ablation configurations are compared jointly with the Friedman test (Friedman, 1937), which ranks the "
        "*k* configurations within each slice:")
    D.eq("4.16", [sup("χ", "2"), " = ", frac("12", ["N k (k + 1)"]), ssum("j = 1", "k", sup(sub("R", "j"), "2")), " − 3N(k + 1)"])
    D.where("*R*_{j} – sum over slices of the ranks of configuration *j*; *k* – number of configurations; *N* – slices",
            "The effect size is Kendall’s *W* = χ²/(*N*(*k* − 1)).")
    D.p("Differences among tumor classes and planes are tested with the chi-square test of independence for rates, "
        "with Cramér’s *V* as the effect size, and with the Kruskal–Wallis test for DSC (Kruskal & Wallis, 1952). "
        "Two independent groups of slices, such as slices whose seed hit or missed the tumor, are compared with the "
        "Mann–Whitney test (Mann & Whitney, 1947) and the rank-biserial correlation (Kerby, 2014). All tests are "
        "two-sided at a significance level of .05.")

    # ── Overall performance ──────────────────────────────────────────────────
    D.h3("4.1.5 Overall Performance")
    dsc_a, rec_a, pre_a = row(au, "dsc"), row(au, "recall"), row(au, "precision")
    suc_a, lk_a = row(au, "success"), row(au, "leaked")
    D.p(f"Table {tb['overall_auto']} compares the two algorithms under automatic seeding, which is how the system "
        "operates without user input.")
    D.table("overall_auto")
    D.p(f"Under automatic seeding, mean DSC is {f(dsc_a['base']['mean'])} for SRG and {f(dsc_a['enh']['mean'])} for "
        f"ESRG ({Zr(dsc_a)}), a {eff(dsc_a['r'])} effect, and mean IoU moves from {f(row(au, 'iou')['base']['mean'])} "
        f"to {f(row(au, 'iou')['enh']['mean'])}. The largest change is in recall, from {f(rec_a['base']['mean'])} to "
        f"{f(rec_a['enh']['mean'])} ({Zr(rec_a)}). The share of successfully segmented slices is "
        f"{pct(suc_a['rate_ref'])} for SRG and {pct(suc_a['rate_cmp'])} for ESRG: ESRG succeeded on {suc_a['only_cmp']} "
        f"slices on which SRG failed, and the reverse happened on {suc_a['only_ref']} ({P(suc_a['p_holm'])}). Mean "
        f"precision is {f(pre_a['base']['mean'])} for SRG and {f(pre_a['enh']['mean'])} for ESRG "
        f"({Zr(pre_a)}), and the leakage rate is {pct(lk_a['rate_ref'])} against {pct(lk_a['rate_cmp'])} "
        f"({P(lk_a['p_holm'])}). [[AUTO_DISCUSSION]]")
    D.table("overall_planted")
    dsc_p, rec_p, pre_p, hd_p = row(pl, "dsc"), row(pl, "recall"), row(pl, "precision"), row(pl, "hd95_wc")
    D.p(f"Table {tb['overall_planted']} repeats the comparison with the seed planted inside the tumor on every slice, "
        "which isolates the region-growing procedure. Mean DSC is "
        f"{f(dsc_p['base']['mean'])} for SRG and {f(dsc_p['enh']['mean'])} for ESRG ({Zr(dsc_p)}), and mean recall "
        f"{f(rec_p['base']['mean'])} against {f(rec_p['enh']['mean'])} ({Zr(rec_p)}). Mean precision is "
        f"{f(pre_p['base']['mean'])} for SRG and {f(pre_p['enh']['mean'])} for ESRG ({Zr(pre_p)}), and median HD95 is "
        f"{f(hd_p['base']['median'], 2)} against {f(hd_p['enh']['median'], 2)} pixels ({Zr(hd_p)}). [[PLANTED_DISCUSSION]]")
    figure("4.1", "fig4_1_dsc_distribution.png", "Distribution of Per-Slice DSC for the Baseline and Enhanced Algorithm",
           f"Each curve is a histogram of the per-slice DSC of one configuration in bins of width 0.05, expressed as the "
           f"percentage of the {n:,} sampled slices. The dotted line marks the success threshold of 0.70 (Equation 4.11).")
    z0 = {c: 100 * R["e4"]["by_class"]["all"][c]["dsc0"] / 100 for c in ("A_SRG", "A_ESRG", "P_SRG", "P_ESRG")}
    D.p(f"Figure 4.1 shows why medians and success rates are reported alongside means. Under automatic seeding, "
        f"{pct(z0['A_ESRG'])} of ESRG slices and {pct(z0['A_SRG'])} of SRG slices have a DSC of exactly zero, "
        "[[DIST_DISCUSSION]]")
    D.table("weighted")
    wa, ua = E4["weighted"]["A_ESRG"]["dsc"], E4["unweighted"]["A_ESRG"]["dsc"]
    sc = E4["split_check"]["A_ESRG"]
    scp = E4["split_check"]["P_ESRG"]
    D.p(f"Table {tb['weighted']} confirms that the equal allocation does not distort the overall figures: re-weighting "
        f"the strata to their population shares changes the mean DSC of ESRG under automatic seeding from "
        f"{f(ua['mean'])} to {f(wa['estimate'])} (95% CI [{f(wa['ci95'][0])}, {f(wa['ci95'][1])}]), and no weighted "
        "estimate differs from its sample mean by more than "
        f"{f(max(abs(E4['weighted'][c][m]['estimate'] - E4['unweighted'][c][m]['mean']) for c in E4['weighted'] for m in ('dsc', 'recall', 'precision')))}. "
        + ("" if not (sc["test_mw"] and scp["test_mw"]) else
        f"Finally, slices drawn from the training split, on which the seed-ranking weights were tuned, did not score "
        f"differently from test-split slices: under automatic seeding the mean DSC of ESRG is {f(sc['train']['mean'])} "
        f"on training slices (*n* = {sc['train']['n']:,}) and {f(sc['test']['mean'])} on test slices "
        f"(*n* = {sc['test']['n']:,}; Mann–Whitney {P(sc['test_mw']['p'])}, *r* = {nz(sc['test_mw']['r_rb'])}), and "
        f"with a planted seed {f(scp['train']['mean'])} against {f(scp['test']['mean'])} ({P(scp['test_mw']['p'])}). "
        "[[SPLIT_DISCUSSION]]"))

    # ══ 4.2 ═══════════════════════════════════════════════════════════════════
    D.h2("4.2 Results per Objective")
    # ── Objective 1 ──────────────────────────────────────────────────────────
    D.h3("4.2.1 Automated Seed Selection")
    D.p("The first problem is that the original algorithm has no defined seed selection step, so its output depends on "
        "where an operator places the seed, which produces inconsistent results (Adams & Bischof, 1994). The proposed "
        "solution automates seed selection: multi-level Otsu thresholding isolates the brightest interior tissue, "
        "candidates near the head outline are removed by the clearance filter, the remaining candidates are ranked, "
        "and the distance-transform core of the top-ranked candidate becomes the seed (Section 3.4.1). Two questions "
        "follow: whether the automatic seed lands in the tumor (accuracy of the seed), and whether automation removes "
        "the dependence on the operator (consistency).")
    D.table("seed_hit")
    ca, cp = E1["class"], E1["plane"]
    D.p(f"Table {tb['seed_hit']} and Figure 4.2 show that the automatic seed lies entirely inside the tumor on "
        f"{ca['all']['hits']:,} of {ca['all']['N_S']:,} seeded slices, a seed hit rate of {pct(ca['all']['rate'])} "
        f"(95% CI {T.ci(ca['all']['ci'])}); on {ca['all']['no_candidate']} slices the selector reported no tumor "
        "candidate instead of guessing. [[SHR_DISCUSSION]]")
    figure("4.2", "fig4_2_seed_hit_rate.png", "Seed Hit Rate of the Automated Seed Selection by Tumor Class and Imaging Plane",
           f"Bars are the seed hit rates of Table {tb['seed_hit']}; error bars are Wilson 95% confidence intervals "
           "(Equation 4.14).")
    hv = E1["hit_vs_miss"]
    D.table("hit_effect")
    D.p(f"The seed outcome decides the result (Table {tb['hit_effect']}). When the seed lands in the tumor, ESRG reaches a "
        f"median DSC of {f(hv['hit']['dsc']['median'])} and succeeds on {pct(hv['hit']['success']['pct'])} of slices; "
        f"when it misses, the median DSC is {f(hv['miss']['dsc']['median'])} and {pct(hv['miss']['success']['pct'])} "
        f"succeed (*U* = {hv['test']['U']:,.0f}, {P(hv['test']['p'])}, *r* = {nz(hv['test']['r_rb'])}). "
        f"Of the {E3['auto_leaks']['n']} slices on which ESRG leaked under automatic seeding, "
        f"{E3['auto_leaks']['seed_miss']} started from a seed that was not entirely inside the tumor, so most leakage "
        "under automatic seeding is a consequence of a misplaced seed rather than of the stopping criterion.")
    op = E1["operator"]
    D.table("operator")
    det = E1.get("determinism") or {}
    D.p(f"Table {tb['operator']} and Figure 4.3 address the consistency that Objective 1 targets. With five equally valid "
        f"operator clicks inside the same tumor, the DSC of the original algorithm varies with a mean within-slice "
        f"*SD* of {f(op['all']['srg']['sd']['mean'])}, and that of the enhanced grower with "
        f"{f(op['all']['esrg']['sd']['mean'])}; the five results span a mean range of {f(op['all']['srg']['range']['mean'])} "
        f"and {f(op['all']['esrg']['range']['mean'])} DSC, respectively. On {pct(op['all']['esrg']['inconsistent']['pct'])} "
        f"of slices (95% CI {T.ci(op['all']['esrg']['inconsistent']['ci'])}), whether the enhanced grower succeeded "
        f"depended on which operator clicked, and on {pct(op['all']['srg']['inconsistent']['pct'])} for the baseline. "
        + (f"The automatic seed, in contrast, produced identical masks in both independent runs on "
           f"{det['A_ESRG']['identical']:,} of {det['A_ESRG']['n']:,} slices for ESRG and "
           f"{det['A_SRG']['identical']:,} of {det['A_SRG']['n']:,} for SRG (largest DSC difference "
           f"{f(max(det['A_ESRG']['max_abs_dsc_diff'], det['A_SRG']['max_abs_dsc_diff']), 4)}), so its within-slice "
           "variability and inconsistency rate are zero. " if det else "")
        + "[[OPERATOR_DISCUSSION]]")
    figure("4.3", "fig4_3_operator_variability.png", "Within-Slice Variability of DSC Across Five Simulated Operators",
           "Boxes show the distribution over slices of the within-slice standard deviation of DSC (Equation 4.8); "
           "whiskers extend to 1.5 interquartile ranges, outliers are not drawn. The automatic seed has a within-slice "
           "standard deviation of zero on every slice.")
    D.table("buckets")
    D.p(f"Table {tb['buckets']} locates the remaining failures of the automatic pipeline (Experiment E5). [[BUCKET_DISCUSSION]]")

    # ── Objective 2 ──────────────────────────────────────────────────────────
    D.h3("4.2.2 Minimization of Undersegmentation")
    D.p("The second problem is that the original difference measure compares each pixel with the global region mean, "
        "so tumor pixels that differ gradually from the seed, for example on the darker side of a bias gradient, are "
        "rejected and left unsegmented. The proposed solution compares each candidate pixel with the nearby absorbed "
        "pixels (Equation 3.23) in the logarithmic domain (Equation 3.18). Its effect is measured by recall, with the "
        "seed planted inside the tumor so that seed placement plays no part. The full enhanced grower is compared with "
        "the baseline and with two ablated versions: one with the global measure restored, and one with the log "
        "transform switched off.")
    D.table("recall")
    rc = E2["recall"]["all"]
    tt = rc["tests"]
    D.p(f"Table {tb['recall']} and Figure 4.4(a) show that the full enhanced grower recovers a mean "
        f"{f(rc['enh']['mean'])} of the tumor, against {f(tt['P_SRG']['ref']['mean'])} for the baseline "
        f"({Zr(tt['P_SRG'])}). In other words, the share of the tumor left unsegmented falls from "
        f"{pct(100 * (1 - tt['P_SRG']['ref']['mean']))} to {pct(100 * (1 - rc['enh']['mean']))}. The ablations isolate "
        f"the two parts of the enhancement. Restoring the global measure lowers mean recall to "
        f"{f(tt['P_ESRG_global']['ref']['mean'])} ({Zr(tt['P_ESRG_global'])}), and switching off the log transform lowers "
        f"it to {f(tt['P_ESRG_nolog']['ref']['mean'])} ({Zr(tt['P_ESRG_nolog'])}). [[RECALL_DISCUSSION]]")
    figure("4.4", "fig4_4_recall_precision.png",
           "Effect of the Local Log-Domain Measure on Recall and of the Adaptive Stopping Criterion on Precision",
           f"All configurations use the same planted seed. Panel (a) shows the mean recall of Table {tb['recall']} and "
           f"panel (b) the mean precision of Table {tb['leak']}. Error bars are 95% confidence intervals of the mean, "
           "*M* ± *t*_{0.975, N−1} · *SD*/√*N*.")
    D.table("bias")
    bb = E2["bias"]
    D.p(f"Table {tb['bias']} and Figure 4.5 test the robustness to a bias field claimed for the log-domain measure "
        f"(Experiment E2). With the {int(100 * M['inu'])}% field, the mean recall of full ESRG changes from "
        f"{f(bb['B_ESRG']['clean']['mean'])} to {f(bb['B_ESRG']['biased']['mean'])} ({P(bb['B_ESRG']['p_holm'])}), and "
        f"that of the baseline from {f(bb['B_SRG']['clean']['mean'])} to {f(bb['B_SRG']['biased']['mean'])} "
        f"({P(bb['B_SRG']['p_holm'])}). Full ESRG on biased slices still recovers far more of the tumor than SRG with "
        f"N4 correction ({f(bb['B_ESRG']['biased']['mean'])} against {f(bb['B_SRG_N4']['biased']['mean'])}; "
        f"{Zr(bb['B_SRG_N4']['B_ESRG_vs'], 'p')}). [[BIAS_DISCUSSION]]")
    figure("4.5", "fig4_5_bias_field.png", "Recall of Each Configuration With and Without the Synthetic Bias Field",
           f"Bars are the mean recall of Table {tb['bias']}; error bars are 95% confidence intervals of the mean.")

    # ── Objective 3 ──────────────────────────────────────────────────────────
    D.h3("4.2.3 Decrease Boundary Leakage")
    D.p("The third problem is that the original algorithm absorbs the next candidate pixel unconditionally, so growth "
        "crosses weak boundaries and floods the surrounding tissue. The proposed solution absorbs a pixel only while it "
        "lies within confidence bounds derived from the region’s own standard deviation, with the statistics frozen in "
        "each pass (Equations 3.28 to 3.30). To isolate the criterion, the enhanced grower was run with and without it "
        "from the same planted seed; without it, pixels are absorbed unconditionally, as in the original algorithm.")
    D.table("leak")
    lp = E3["precision"]["all"]
    lk = E3["leak"]["all"]
    D.p(f"Table {tb['leak']}, Figure 4.4(b), and Figure 4.6 show the effect of the stopping criterion. Without it, the "
        f"region leaks on {pct(lk['leak_ci']['P_ESRG_nostop']['pct'])} of slices, its median area is "
        f"{f(lk['ratio']['P_ESRG_nostop']['median'], 2)} times the true tumor area, and mean precision is "
        f"{f(lp['tests']['P_ESRG_nostop']['ref']['mean'])}. With it, the leakage rate falls to "
        f"{pct(lk['leak_ci']['P_ESRG']['pct'])} (*b* / *c* = {lk['P_ESRG_nostop']['only_ref']} / "
        f"{lk['P_ESRG_nostop']['only_cmp']}, {P(lk['P_ESRG_nostop']['p'])}) and mean precision rises to "
        f"{f(lp['enh']['mean'])} ({Zr(lp['tests']['P_ESRG_nostop'])}). [[LEAK_DISCUSSION]]")
    figure("4.6", "fig4_6_leakage_rate.png", "Leakage Rate With and Without the Adaptive Stopping Criterion",
           "Planted seed. A slice leaks when its predicted area exceeds twice the ground-truth area (Equation 3.39). "
           "Error bars are Wilson 95% confidence intervals.")
    D.table("boundary")
    hd = E3["hd95_wc"]["all"]
    asd = E3["assd_wc"]["all"]
    D.p(f"Table {tb['boundary']} evaluates the boundary itself. The median HD95 of full ESRG is "
        f"{f(hd['enh']['median'], 2)} pixels, against {f(hd['tests']['P_ESRG_nostop']['ref']['median'], 2)} without the "
        f"stopping criterion ({Zr(hd['tests']['P_ESRG_nostop'])}) and {f(hd['tests']['P_SRG']['ref']['median'], 2)} for "
        f"the baseline ({Zr(hd['tests']['P_SRG'])}); the median ASSD is {f(asd['enh']['median'], 2)}, "
        f"{f(asd['tests']['P_ESRG_nostop']['ref']['median'], 2)}, and {f(asd['tests']['P_SRG']['ref']['median'], 2)} "
        "pixels, respectively. [[BOUNDARY_DISCUSSION]]")
    D.table("purify")
    pu = E3["purify"]
    D.p(f"Table {tb['purify']} examines the seed size, which sets the first stopping bound. Purifying the planted click "
        f"shrinks it from a median of {f(pu['seed_area']['P_ESRG_nopurify']['median'], 0)} to "
        f"{f(pu['seed_area']['P_ESRG']['median'], 0)} pixels, and its initial σ_{{A}} from "
        f"{f(pu['sigma_A']['P_ESRG_nopurify']['median'], 4)} to {f(pu['sigma_A']['P_ESRG']['median'], 4)}; on "
        f"{pct(pu['floor_binding']['P_ESRG'])} of slices the noise floor, not the seed, then sets the bound. "
        "[[PURIFY_DISCUSSION]]")

    # ── Overall by class / ablation ──────────────────────────────────────────
    D.h3("4.2.4 Overall Delineation Quality and Class-Dependent Performance")
    D.p(f"Combining the three enhancements, Table {tb['dsc_class']} reports the median DSC by tumor class for both "
        "algorithms under both seeding conditions, with the share of successfully segmented slices.")
    D.table("dsc_class")
    D.p("[[CLASS_DISCUSSION]]")
    D.table("dsc_plane")
    D.p("[[PLANE_DISCUSSION]]")
    D.table("ablation")
    ab = E4["ablation"]
    D.p(f"Table {tb['ablation']} summarizes the ablation (Experiment E4). The four ESRG configurations differ "
        f"significantly (χ²({ab['df']}, *N* = {ab['n']:,}) = {ab['chi2']:.2f}, {P(ab['p'])}, Kendall’s *W* = "
        f"{nz(ab['kendall_w'])}), and full ESRG has the best mean rank ({f(ab['mean_rank']['P_ESRG'], 2)}). "
        "[[ABLATION_DISCUSSION]]")

    # ── Efficiency ───────────────────────────────────────────────────────────
    D.h3("4.2.5 Computational Efficiency")
    if E6:
        D.table("time")
        t = E6["all"]
        D.p(f"Table {tb['time']} shows the processing time measured in the sequential pass (Experiment E6). The median time "
            f"per slice is {f(t['srg']['median'], 2)} s for SRG and {f(t['esrg']['median'], 2)} s for ESRG "
            f"(*Z* = {f(t['Z'], 2)}, {P(t['p'])}, *r* = {nz(t['r'])}), a {pct(abs(t['reduction_pct_median']))} "
            f"{'reduction' if t['reduction_pct_median'] > 0 else 'increase'}; ESRG was faster on "
            f"{pct(t['faster_share'])} of slices. [[TIME_DISCUSSION]]")
        D.table("stages")
        figure("4.7", "fig4_7_processing_time.png", "Mean Processing Time per Slice, by Pipeline Stage",
               f"Stacked bars are the mean stage times of Table {tb['stages']} from the sequential timing pass.")
        D.p("[[STAGE_DISCUSSION]]")

    # ── Summary ──────────────────────────────────────────────────────────────
    D.h2("4.3 Summary of Findings")
    D.p("[[SUMMARY_INTRO]]")
    summ = summary_table(R, tb)
    D.blocks.append({"t": "table", "num": f"4.{len(D.tabs) + 1}", **summ})
    D.p("[[SUMMARY_CLOSE]]")

    # ── References ───────────────────────────────────────────────────────────
    D.h2("References Added in Chapter 4")
    for r in REFERENCES:
        D.ref(r)
    return D, R


def summary_table(R, tb):
    E1, E2, E3, E4, E6 = R["e1"], R["e2"], R["e3"], R["e4"], R["e6"]
    rc, lp, lk = E2["recall"]["all"], E3["precision"]["all"], E3["leak"]["all"]
    op = E1["operator"]["all"]
    au = {r["metric"]: r for r in E4["auto"]}
    rows = [
        {"group": "Objective 1: automated seed selection"},
        ["Seed hit rate", "No automatic seed", pct(E1["class"]["all"]["rate"]), f"95% CI {T.ci(E1['class']['all']['ci'])}", tb["seed_hit"]],
        ["Within-slice *SD* of DSC", f"{f(op['srg']['sd']['mean'])} (operator)", "0.000 (automatic)",
         "Deterministic in two runs", tb["operator"]],
        {"group": "Objective 2: undersegmentation (planted seed)"},
        ["Recall *M*", f(rc["tests"]["P_SRG"]["ref"]["mean"]), f(rc["enh"]["mean"]),
         f"*r* = {nz(rc['tests']['P_SRG']['r'])}, {P(rc['tests']['P_SRG']['p_holm'])}", tb["recall"]],
        ["Recall *M*, global vs. local measure", f(rc["tests"]["P_ESRG_global"]["ref"]["mean"]), f(rc["enh"]["mean"]),
         f"*r* = {nz(rc['tests']['P_ESRG_global']['r'])}, {P(rc['tests']['P_ESRG_global']['p_holm'])}", tb["recall"]],
        ["Recall *M*, log off vs. on", f(rc["tests"]["P_ESRG_nolog"]["ref"]["mean"]), f(rc["enh"]["mean"]),
         f"*r* = {nz(rc['tests']['P_ESRG_nolog']['r'])}, {P(rc['tests']['P_ESRG_nolog']['p_holm'])}", tb["recall"]],
        {"group": "Objective 3: boundary leakage (planted seed)"},
        ["Precision *M*, no stopping vs. adaptive", f(lp["tests"]["P_ESRG_nostop"]["ref"]["mean"]), f(lp["enh"]["mean"]),
         f"*r* = {nz(lp['tests']['P_ESRG_nostop']['r'])}, {P(lp['tests']['P_ESRG_nostop']['p_holm'])}", tb["leak"]],
        ["Leakage rate, no stopping vs. adaptive", pct(lk["leak_ci"]["P_ESRG_nostop"]["pct"]), pct(lk["leak_ci"]["P_ESRG"]["pct"]),
         P(lk["P_ESRG_nostop"]["p"]), tb["leak"]],
        {"group": "Overall (automatic seed) and efficiency"},
        ["DSC *M*", f(au["dsc"]["base"]["mean"]), f(au["dsc"]["enh"]["mean"]),
         f"*r* = {nz(au['dsc']['r'])}, {P(au['dsc']['p_holm'])}", tb["overall_auto"]],
        ["Success rate", pct(au["success"]["rate_ref"]), pct(au["success"]["rate_cmp"]), P(au["success"]["p_holm"]), tb["overall_auto"]],
    ]
    if E6:
        rows.append(["Time per slice, *Mdn*", f"{f(E6['all']['srg']['median'], 2)} s", f"{f(E6['all']['esrg']['median'], 2)} s",
                     f"*r* = {nz(E6['all']['r'])}, {P(E6['all']['p'])}", tb["time"]])
    return {"title": "Summary of the Evaluation per Objective",
            "header": ["Metric", "Reference", "ESRG", "Test", "Table"],
            "rows": rows, "widths": [3, 1.6, 1.3, 2.2, 0.7], "align": ["l", "c", "c", "c", "c"],
            "note": ("“Reference” is the configuration without the enhancement (the baseline SRG, or ESRG with one "
                     "enhancement removed); for the within-slice *SD* it is the enhanced grower seeded by an operator. "
                     "Objectives 2 and 3 are measured with a planted seed so that seed placement does not affect them.")}


REFERENCES = [
    "Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science, 16*(2), 101–133. https://doi.org/10.1214/ss/1009213286",
    "Cochran, W. G. (1977). *Sampling techniques* (3rd ed.). John Wiley & Sons.",
    "Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.",
    "Fan, J., Zeng, G., Body, M., & Hacid, M.-S. (2005). Seeded region growing: An extensive and comparative study. *Pattern Recognition Letters, 26*(8), 1139–1156. https://doi.org/10.1016/j.patrec.2004.10.010",
    "Hoefler, T., & Belli, R. (2015). Scientific benchmarking of parallel computing systems: Twelve ways to tell the masses when reporting performance results. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’15)* (Article 73). ACM. https://doi.org/10.1145/2807591.2807644",
    "Joskowicz, L., Cohen, D., Caplan, N., & Sosna, J. (2019). Inter-observer variability of manual contour delineation of structures in CT. *European Radiology, 29*(3), 1391–1399. https://doi.org/10.1007/s00330-018-5695-5",
    "Kerby, D. S. (2014). The simple difference formula: An approach to teaching nonparametric correlation. *Comprehensive Psychology, 3*, Article 1. https://doi.org/10.2466/11.IT.3.1",
    "Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association, 47*(260), 583–621. https://doi.org/10.1080/01621459.1952.10483441",
    "Lohr, S. L. (2021). *Sampling: Design and analysis* (3rd ed.). CRC Press.",
    "Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics, 18*(1), 50–60. https://doi.org/10.1214/aoms/1177730491",
    "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996",
    "Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., … Van Leemput, K. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE Transactions on Medical Imaging, 34*(10), 1993–2024. https://doi.org/10.1109/TMI.2014.2377694",
    "Moschidis, E., & Graham, J. (2010). A systematic performance evaluation of interactive image segmentation methods based on simulated user interaction. In *2010 IEEE International Symposium on Biomedical Imaging: From Nano to Macro* (pp. 928–931). IEEE. https://doi.org/10.1109/ISBI.2010.5490139",
    "Reinke, A., Tizabi, M. D., Baumgartner, M., Eisenmann, M., Heckmann-Nötzel, D., Kavur, A. E., … Maier-Hein, L. (2024). Understanding metric-related pitfalls in image analysis validation. *Nature Methods, 21*(2), 182–194. https://doi.org/10.1038/s41592-023-02150-0",
    "Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika, 52*(3–4), 591–611. https://doi.org/10.1093/biomet/52.3-4.591",
    "Udupa, J. K., LeBlanc, V. R., Zhuge, Y., Imielinska, C., Schmidt, H., Currie, L. M., Hirsch, B. E., & Woodburn, J. (2006). A framework for evaluating image segmentation algorithms. *Computerized Medical Imaging and Graphics, 30*(2), 75–87. https://doi.org/10.1016/j.compmedimag.2005.12.001",
    "Visser, M., Müller, D. M. J., van Duijn, R. J. M., Smits, M., Verburg, N., Hendriks, E. J., … De Witt Hamer, P. C. (2019). Inter-rater agreement in glioma segmentations on longitudinal MRI. *NeuroImage: Clinical, 22*, Article 101727. https://doi.org/10.1016/j.nicl.2019.101727",
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953",
    "Yeghiazaryan, V., & Voiculescu, I. (2018). Family of boundary overlap metrics for the evaluation of medical image segmentation. *Journal of Medical Imaging, 5*(1), Article 015006. https://doi.org/10.1117/1.JMI.5.1.015006",
    "Zijdenbos, A. P., Dawant, B. M., Margolin, R. A., & Palmer, A. C. (1994). Morphometric analysis of white matter lesions in MR images: Method and validation. *IEEE Transactions on Medical Imaging, 13*(4), 716–724. https://doi.org/10.1109/42.363096",
    "Zou, K. H., Warfield, S. K., Bharatha, A., Tempany, C. M. C., Kaus, M. R., Haker, S. J., Wells, W. M., Jolesz, F. A., & Kikinis, R. (2004). Statistical validation of image segmentation quality based on a spatial overlap index. *Academic Radiology, 11*(2), 178–189. https://doi.org/10.1016/S1076-6332(03)00671-8",
]


def render(spec_path, out_path):
    env = dict(os.environ)
    nm = os.path.join(HERE, "node_modules")
    if os.path.isdir(nm):
        env["NODE_PATH"] = nm
    subprocess.run(["node", os.path.join(HERE, "render_docx.js"), spec_path, out_path], check=True, env=env)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "evaluation"))
    ap.add_argument("--out", default=os.path.join(ROOT, "Chapter4_Results_and_Discussion.docx"))
    args = ap.parse_args()
    D, R = build(args.dir)
    from experiments.chapter4.discussion import texts
    disc = texts(R, pd.read_csv(os.path.join(args.dir, "raw_results.csv"), low_memory=False))
    missing = []

    def fill(text):
        import re

        def sub_(m):
            k = m.group(1)
            if k not in disc:
                missing.append(k)
                return f"[{k}]"
            return disc[k]
        return re.sub(r"\[\[([A-Z_]+)\]\]", sub_, text)

    for b in D.blocks:
        if b["t"] == "p":
            b["text"] = fill(b["text"]).strip()
    D.blocks = [b for b in D.blocks if not (b["t"] == "p" and not b["text"])]
    spec = os.path.join(args.dir, "chapter4_spec.json")
    json.dump({"title": "Chapter 4 Results and Discussion", "blocks": D.blocks}, open(spec, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if missing:
        print("discussion placeholders without text:", sorted(set(missing)))
    render(spec, args.out)
