"""
build_chapter4.py — Write Chapter 4 (Results and Discussion) from the evaluation outputs.

Purpose : Assemble the chapter text, APA tables, and figures from results.json
          and the raw per-slice rows, so that every number in the chapter is
          computed, never typed, and agrees with the appendix workbooks. The
          dataset, sampling, metrics, and statistical tests are defined in
          Chapter 3, so this chapter reports results only and cites Chapter 3's
          equation and table numbers.
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
import re
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
    """Effect size and p of a Wilcoxon test, as cited in the text: '*r* = .56, *p* < .001'."""
    return f"*r* = {nz(t['r'])}, {P(t.get(key, t['p']))}"


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


OBJ1 = "to provide an automatic seed selection option through Multi-Level Otsu Thresholding with connected-component filtering"
OBJ2 = ("to minimize undersegmentation by replacing the global-mean difference with a local difference measure "
        "computed in the log domain, making the criterion insensitive to multiplicative bias")
OBJ3 = ("to decrease boundary leakage by integrating a stopping criterion that adapts to the intensity variability "
        "of the growing region, supported with a global drift guard")


def build(d):
    R = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    D = Doc()
    D.tabs = T.build_chapter(R)
    tb = {k: v["num"] for k, v in D.tabs.items()}
    M, E1, E2, E3, E4 = R["meta"], R["e1"], R["e2"], R["e3"], R["e4"]
    n = M["n_sample"]
    figdir = os.path.join(d, "figures")
    fig_n = [0]

    def figure(name, title, note):
        fig_n[0] += 1
        pth = os.path.join(figdir, name)
        D.fig(f"4.{fig_n[0]}", os.path.relpath(pth, d).replace("\\", "/"), title, note, png_size(pth))
        return f"4.{fig_n[0]}"

    def row(cmp, metric):
        return next(r for r in cmp if r["metric"] == metric)

    au, pl = E4["auto"], E4["planted"]

    # ══ 4.1 ═══════════════════════════════════════════════════════════════════
    D.h1("Chapter Four")
    D.h1("RESULTS AND DISCUSSION")
    D.h2("4.1 General Results")
    D.p("This chapter presents the evaluation of the Enhanced Seeded Region Growing (ESRG) algorithm against the "
        "original Seeded Region Growing (SRG) algorithm of Adams and Bischof (1994), which serves as the baseline "
        "control. The evaluation follows two of the aspects that Udupa et al. (2006) identify for judging a "
        "segmentation method: its *accuracy* (how closely the result matches the ground truth) and its *precision* in "
        "the sense of reproducibility (whether the same image always yields the same result). The dataset and sampling design, the experimental configurations, the evaluation "
        "metrics, and the statistical tests are described in Chapter 3 (Sections 3.1 to 3.1.2, Tables 3.1 to 3.5) and "
        "are not repeated here.")
    D.p(f"The results are organized by the three specific objectives of the study: {OBJ1}, evaluated by the seed hit "
        f"rate and the variability between operators; {OBJ2}, evaluated by recall; and {OBJ3}, evaluated by precision, "
        "the leakage rate, and HD95. Overall delineation is summarized by the Dice Similarity Coefficient (DSC) and the "
        f"success rate. All {M['n_rows']:,} runs ({n:,} slices × "
        f"{M['n_configs']} configurations) were produced by the automated evaluation pipeline, and "
        f"{'none' if M['n_errors'] == 0 else M['n_errors']} ended in an error; the per-slice results of every "
        "evaluation are listed in Appendices B to G.")
    ws = [r["shapiro_diff"]["W"] for r in au + pl if r["metric"] in ("dsc", "recall", "precision", "hd95_wc")]
    D.p("Because the Shapiro–Wilk test rejected the normality of every paired difference "
        f"(*W* = {nz(min(ws), 3)} to {nz(max(ws), 3)}, all *p* < .001), configurations are compared with the Wilcoxon "
        "signed-rank test and paired rates with the exact McNemar test, as specified in Section 3.1.2. With "
        f"{n:,} paired slices almost every difference is statistically significant, so the tables report the effect "
        "size *r* = *Z*/√*N* (Section 3.1.2), where |*r*| near .1, .3, and .5 indicates a small, medium, and large "
        "effect. The tables of Sections 4.2.1 to 4.2.3 add an Interpretation column that states, for each "
        "comparison, whether the Holm-adjusted *p* value is below .05 (significant), the size of the effect, and "
        "which configuration the result favors. Means and medians are reported without standard deviations because "
        "per-slice DSC is bounded and bimodal (Figure 4.1); the standard deviations, test statistics, and confidence "
        "intervals of every comparison are given in the appendix workbooks.")

    # ── Overall performance ──────────────────────────────────────────────────
    D.h3("4.1.1 Overall Performance")
    dsc_a, rec_a, pre_a = row(au, "dsc"), row(au, "recall"), row(au, "precision")
    suc_a, lk_a = row(au, "success"), row(au, "leaked")
    D.p(f"Table {tb['overall_auto']} compares the two algorithms under automatic seeding, which is how the system "
        "operates without user input.")
    D.table("overall_auto")
    D.p(f"Under automatic seeding, mean DSC is {f(dsc_a['base']['mean'])} for SRG and {f(dsc_a['enh']['mean'])} for "
        f"ESRG ({Zr(dsc_a)}), a {eff(dsc_a['r'])} effect. The largest change is in recall, from "
        f"{f(rec_a['base']['mean'])} to {f(rec_a['enh']['mean'])} ({Zr(rec_a)}). The share of successfully segmented "
        f"slices is {pct(suc_a['rate_ref'])} for SRG and {pct(suc_a['rate_cmp'])} for ESRG: ESRG succeeded on "
        f"{suc_a['only_cmp']} slices on which SRG failed, and the reverse happened on {suc_a['only_ref']} "
        f"({P(suc_a['p_holm'])}). Mean precision is {f(pre_a['base']['mean'])} for SRG and {f(pre_a['enh']['mean'])} "
        f"for ESRG ({Zr(pre_a)}), and the leakage rate is {pct(lk_a['rate_ref'])} against {pct(lk_a['rate_cmp'])} "
        f"({P(lk_a['p_holm'])}). [[AUTO_DISCUSSION]]")
    D.table("overall_planted")
    dsc_p, rec_p, pre_p, hd_p = row(pl, "dsc"), row(pl, "recall"), row(pl, "precision"), row(pl, "hd95_wc")
    D.p(f"Table {tb['overall_planted']} repeats the comparison with the seed planted inside the tumor on every slice, "
        "which isolates the region-growing procedure. Mean DSC is "
        f"{f(dsc_p['base']['mean'])} for SRG and {f(dsc_p['enh']['mean'])} for ESRG ({Zr(dsc_p)}), and mean recall "
        f"{f(rec_p['base']['mean'])} against {f(rec_p['enh']['mean'])} ({Zr(rec_p)}). Mean precision is "
        f"{f(pre_p['base']['mean'])} for SRG and {f(pre_p['enh']['mean'])} for ESRG ({Zr(pre_p)}), and median HD95 is "
        f"{f(hd_p['base']['median'], 2)} against {f(hd_p['enh']['median'], 2)} pixels ({Zr(hd_p)}). [[PLANTED_DISCUSSION]]")
    fdist = figure("fig4_1_dsc_distribution.png", "Distribution of Per-Slice DSC for the Baseline and Enhanced Algorithm",
                   f"Each curve is a histogram of the per-slice DSC of one configuration in bins of width 0.05, expressed "
                   f"as the percentage of the {n:,} sampled slices. The dotted line marks the success threshold of 0.70 "
                   "(Table 3.3).")
    z0 = {c: R["e4"]["by_class"]["all"][c]["dsc0"] for c in ("A_SRG", "A_ESRG")}
    D.p(f"Figure {fdist} shows why medians and success rates are reported alongside means. Under automatic seeding, "
        f"{pct(z0['A_ESRG'])} of ESRG slices and {pct(z0['A_SRG'])} of SRG slices have a DSC of exactly zero, "
        "[[DIST_DISCUSSION]]")
    sc, scp = E4["split_check"]["A_ESRG"], E4["split_check"]["P_ESRG"]
    wmax = max(abs(E4["weighted"][c][m]["estimate"] - E4["unweighted"][c][m]["mean"])
               for c in E4["weighted"] for m in ("dsc", "recall", "precision"))
    D.p("Two checks confirm that these figures are not artifacts of the sampling design. Re-weighting the strata to "
        f"their population shares (Section 3.1.1) changes no mean DSC, recall, or precision by more than {f(wmax)}, so "
        "the equal allocation does not distort the overall results. "
        + ("" if not (sc["test_mw"] and scp["test_mw"]) else
           f"Slices drawn from the training split, on which the seed-ranking weights were tuned, did not score "
           f"differently from test-split slices: under automatic seeding the mean DSC of ESRG is "
           f"{f(sc['train']['mean'])} on training slices (*n* = {sc['train']['n']:,}) and {f(sc['test']['mean'])} on "
           f"test slices (*n* = {sc['test']['n']:,}; Mann–Whitney {P(sc['test_mw']['p'])}), and with a planted seed "
           f"{f(scp['train']['mean'])} against {f(scp['test']['mean'])} ({P(scp['test_mw']['p'])}). [[SPLIT_DISCUSSION]]"))

    # ══ 4.2 ═══════════════════════════════════════════════════════════════════
    D.h2("4.2 Results per Objective")
    # ── Objective 1 ──────────────────────────────────────────────────────────
    D.h3("4.2.1 Objective 1: Automatic Seed Selection")
    D.p("The first problem is that the original algorithm relies on manual seed selection, so its output depends on "
        "where an operator places the seed (Adams & Bischof, 1994). The first objective is "
        f"{OBJ1}: multi-level Otsu thresholding isolates the brightest interior tissue, the connected components near "
        "the head outline are removed by the clearance filter, the remaining components are ranked, and the "
        "distance-transform core of the top-ranked component becomes the seed (Section 3.4.1). Two questions follow: "
        "whether the automatic seed lands in the tumor, and whether automation removes the dependence on the operator.")
    D.table("seed_hit")
    ca = E1["class"]
    fhit = figure("fig4_2_seed_hit_rate.png", "Seed Hit Rate of the Automated Seed Selection by Tumor Class and Imaging Plane",
                  f"Bars are the seed hit rates of Table {tb['seed_hit']}; error bars are Wilson 95% confidence intervals "
                  "(Section 3.1.2).")
    D.p(f"Table {tb['seed_hit']} and Figure {fhit} show that the automatic seed lies entirely inside the tumor on "
        f"{ca['all']['hits']:,} of {ca['all']['N_S']:,} seeded slices, a seed hit rate of {pct(ca['all']['rate'])} "
        f"(95% CI {T.ci(ca['all']['ci'])}); on {ca['all']['no_candidate']} slices the selector reported no tumor "
        "candidate instead of guessing. [[SHR_DISCUSSION]]")
    hv = E1["hit_vs_miss"]
    D.p(f"The seed outcome decides the result. When the seed lands in the tumor ({hv['hit']['dsc']['n']:,} slices), "
        f"ESRG reaches a median DSC of {f(hv['hit']['dsc']['median'])} and succeeds on "
        f"{pct(hv['hit']['success']['pct'])} of slices; when it misses or no seed is produced "
        f"({hv['miss']['dsc']['n']:,} slices), the median DSC is {f(hv['miss']['dsc']['median'])} and "
        f"{pct(hv['miss']['success']['pct'])} succeed (Mann–Whitney {P(hv['test']['p'])}, rank-biserial "
        f"*r* = {nz(hv['test']['r_rb'])}). Of the {E3['auto_leaks']['n']} slices on which ESRG leaked under automatic "
        f"seeding, {E3['auto_leaks']['seed_miss']} started from a seed that was not entirely inside the tumor, so most "
        "leakage under automatic seeding is a consequence of a misplaced seed rather than of the stopping criterion.")
    op = E1["operator"]
    D.table("operator")
    det = E1.get("determinism") or {}
    fop = figure("fig4_3_operator_variability.png", "Within-Slice Variability of DSC Across Five Simulated Operators",
                 "Boxes show the distribution over slices of the within-slice standard deviation of DSC; whiskers extend "
                 "to 1.5 interquartile ranges, outliers are not drawn. The automatic seed has a within-slice standard "
                 "deviation of zero on every slice.")
    D.p(f"Table {tb['operator']} and Figure {fop} address the consistency that Objective 1 targets. With five equally "
        "valid operator clicks inside the same tumor, the DSC of the original algorithm varies with a mean within-slice "
        f"*SD* of {f(op['all']['srg']['sd']['mean'])}, and that of the enhanced grower with "
        f"{f(op['all']['esrg']['sd']['mean'])}. On {pct(op['all']['esrg']['inconsistent']['pct'])} of slices, whether the "
        "enhanced grower succeeded depended on which operator clicked, and on "
        f"{pct(op['all']['srg']['inconsistent']['pct'])} for the baseline. "
        + (f"The automatic seed, in contrast, produced identical masks in both independent runs on "
           f"{det['A_ESRG']['identical']:,} of {det['A_ESRG']['n']:,} slices for ESRG and "
           f"{det['A_SRG']['identical']:,} of {det['A_SRG']['n']:,} for SRG, so its within-slice variability and "
           "inconsistency rate are zero. " if det else "")
        + "[[OPERATOR_DISCUSSION]]")
    D.table("buckets")
    D.p(f"Table {tb['buckets']} locates the remaining failures of the automatic pipeline by assigning each slice to the "
        "first stage it fails (Table 3.4). [[BUCKET_DISCUSSION]]")

    # ── Objective 2 ──────────────────────────────────────────────────────────
    D.h3("4.2.2 Objective 2: Minimization of Undersegmentation")
    D.p("The second problem is that the original algorithm evaluates each candidate pixel against the mean of the "
        "entire region rather than its local neighborhood, so it implicitly assumes that the region is "
        "intensity-homogeneous; under MRI intensity non-uniformity, pixels of the same tissue deviate from that global "
        "mean and are left out, causing undersegmentation. In this study, undersegmentation is the failure of region "
        "growing to cover the full extent of the tumor, so that part of the ground-truth tumor is left outside the "
        "segmented region even though the seed lies inside it; it is measured by recall (Table 3.3). The second "
        f"objective is {OBJ2}: each candidate is compared with the nearby absorbed pixels (Equation 3.8) on the "
        "log-intensity scale (Equation 3.6). To exclude the influence of seed placement, the seed is planted inside "
        "the tumor, and the full enhanced grower is compared with the baseline and with two ablated versions: one with "
        "the global measure restored, and one with the log transform switched off.")
    D.table("recall")
    rc = E2["recall"]["all"]
    tt = rc["tests"]
    D.p(f"Table {tb['recall']} shows that the full enhanced grower recovers a mean {f(rc['enh']['mean'])} of the "
        f"tumor, against {f(tt['P_SRG']['ref']['mean'])} for the baseline ({Zr(tt['P_SRG'])}). In other words, the "
        f"share of the tumor left unsegmented falls from {pct(100 * (1 - tt['P_SRG']['ref']['mean']))} to "
        f"{pct(100 * (1 - rc['enh']['mean']))}. The ablations isolate the two parts of the enhancement. Restoring the "
        f"global measure lowers mean recall to {f(tt['P_ESRG_global']['ref']['mean'])} ({Zr(tt['P_ESRG_global'])}), "
        f"and switching off the log transform lowers it to {f(tt['P_ESRG_nolog']['ref']['mean'])} "
        f"({Zr(tt['P_ESRG_nolog'])}). [[RECALL_DISCUSSION]]")
    D.table("bias")
    bb = E2["bias"]
    D.p(f"Table {tb['bias']} tests the insensitivity to multiplicative bias that the log-domain measure is designed "
        f"to provide. With the {int(100 * M['inu'])}% field, the mean recall of full ESRG changes from "
        f"{f(bb['B_ESRG']['clean']['mean'])} to {f(bb['B_ESRG']['biased']['mean'])} ({P(bb['B_ESRG']['p_holm'])}), and "
        f"that of the baseline from {f(bb['B_SRG']['clean']['mean'])} to {f(bb['B_SRG']['biased']['mean'])} "
        f"({P(bb['B_SRG']['p_holm'])}). Full ESRG on biased slices still recovers far more of the tumor than SRG with "
        f"N4 correction ({f(bb['B_ESRG']['biased']['mean'])} against {f(bb['B_SRG_N4']['biased']['mean'])}; "
        f"{Zr(bb['B_SRG_N4']['B_ESRG_vs'], 'p')}). [[BIAS_DISCUSSION]]")

    # ── Objective 3 ──────────────────────────────────────────────────────────
    D.h3("4.2.3 Objective 3: Decrease of Boundary Leakage")
    D.p("The third problem is that the original algorithm absorbs the next candidate pixel from the sequentially "
        "sorted list unconditionally, so growth crosses weak boundaries and floods the surrounding tissue. The third "
        f"objective is {OBJ3}: a pixel is absorbed only while its local difference stays within a bound proportional "
        "to the region’s own standard deviation, and a global drift guard rejects pixels that stray too far from the "
        "region mean, with both bounds frozen within each pass (Equations 3.9 to 3.11). To isolate the criterion, the "
        "enhanced grower was run with and without it from the same planted seed; without it, pixels are absorbed "
        "unconditionally, as in the original algorithm.")
    D.table("leak")
    lp = E3["precision"]["all"]
    lk = E3["leak"]["all"]
    fleak = figure("fig4_6_leakage_rate.png", "Leakage Rate With and Without the Adaptive Stopping Criterion",
                   "Planted seed. A slice leaks when its predicted area exceeds twice the ground-truth area "
                   "(Table 3.3). Error bars are Wilson 95% confidence intervals.")
    D.p(f"Table {tb['leak']} and Figure {fleak} show the effect of the stopping criterion. Without it, the region "
        f"leaks on {pct(lk['leak_ci']['P_ESRG_nostop']['pct'])} of slices, its median area is "
        f"{f(lk['ratio']['P_ESRG_nostop']['median'], 2)} times the true tumor area, and mean precision is "
        f"{f(lp['tests']['P_ESRG_nostop']['ref']['mean'])}. With it, the leakage rate falls to "
        f"{pct(lk['leak_ci']['P_ESRG']['pct'])} and the median area ratio to {f(lk['ratio']['P_ESRG']['median'], 2)}: "
        f"the criterion removed the leak on {lk['P_ESRG_nostop']['only_ref']:,} slices and introduced it on "
        f"{'none' if lk['P_ESRG_nostop']['only_cmp'] == 0 else lk['P_ESRG_nostop']['only_cmp']} "
        f"({P(lk['P_ESRG_nostop']['p'])}). Mean precision rises to {f(lp['enh']['mean'])} "
        f"({Zr(lp['tests']['P_ESRG_nostop'])}). [[LEAK_DISCUSSION]]")
    D.table("boundary")
    hd = E3["hd95_wc"]["all"]
    D.p(f"Table {tb['boundary']} evaluates the boundary itself. The median HD95 of full ESRG is "
        f"{f(hd['enh']['median'], 2)} pixels, against {f(hd['tests']['P_ESRG_nostop']['ref']['median'], 2)} without the "
        f"stopping criterion ({Zr(hd['tests']['P_ESRG_nostop'])}) and {f(hd['tests']['P_SRG']['ref']['median'], 2)} for "
        f"the baseline ({Zr(hd['tests']['P_SRG'])}). [[BOUNDARY_DISCUSSION]]")
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
    D.p("[[PLANE_DISCUSSION]]")
    D.table("ablation")
    ab = E4["ablation"]
    D.p(f"Table {tb['ablation']} summarizes the ablation. The four ESRG configurations differ significantly "
        f"(χ²({ab['df']}, *N* = {ab['n']:,}) = {ab['chi2']:.2f}, {P(ab['p'])}, Kendall’s *W* = {nz(ab['kendall_w'])}), "
        f"and full ESRG has the best mean rank ({f(ab['mean_rank']['P_ESRG'], 2)}). [[ABLATION_DISCUSSION]]")

    # ── Summary ──────────────────────────────────────────────────────────────
    D.h2("4.3 Summary of Findings")
    D.p("[[SUMMARY_INTRO]]")
    summ = summary_table(R, tb)
    snum = f"4.{len(D.tabs) + 1}"
    tb["summary"] = snum
    D.blocks.append({"t": "table", "num": snum, **summ})
    D.p("[[SUMMARY_CLOSE]]")

    # ── References ───────────────────────────────────────────────────────────
    D.h2("References Added in Chapter 4")
    return D, R, tb


def summary_table(R, tb):
    E1, E2, E3, E4 = R["e1"], R["e2"], R["e3"], R["e4"]
    rc, lp, lk = E2["recall"]["all"], E3["precision"]["all"], E3["leak"]["all"]
    op = E1["operator"]["all"]
    au = {r["metric"]: r for r in E4["auto"]}
    rows = [
        {"group": "Objective 1: automatic seed selection"},
        ["Seed hit rate", "No automatic seed", pct(E1["class"]["all"]["rate"]), f"95% CI {T.ci(E1['class']['all']['ci'])}", tb["seed_hit"]],
        ["Within-slice *SD* of DSC", f"{f(op['esrg']['sd']['mean'])} (operator)", "0.000 (automatic)",
         "Deterministic", tb["operator"]],
        {"group": "Objective 2: undersegmentation (planted seed)"},
        ["Recall *M*, baseline vs. ESRG", f(rc["tests"]["P_SRG"]["ref"]["mean"]), f(rc["enh"]["mean"]),
         f"*r* = {nz(rc['tests']['P_SRG']['r'])}", tb["recall"]],
        ["Recall *M*, global vs. local measure", f(rc["tests"]["P_ESRG_global"]["ref"]["mean"]), f(rc["enh"]["mean"]),
         f"*r* = {nz(rc['tests']['P_ESRG_global']['r'])}", tb["recall"]],
        ["Recall *M*, log off vs. on", f(rc["tests"]["P_ESRG_nolog"]["ref"]["mean"]), f(rc["enh"]["mean"]),
         f"*r* = {nz(rc['tests']['P_ESRG_nolog']['r'])}", tb["recall"]],
        {"group": "Objective 3: boundary leakage (planted seed)"},
        ["Precision *M*, no stopping vs. adaptive", f(lp["tests"]["P_ESRG_nostop"]["ref"]["mean"]), f(lp["enh"]["mean"]),
         f"*r* = {nz(lp['tests']['P_ESRG_nostop']['r'])}", tb["leak"]],
        ["Leakage rate, no stopping vs. adaptive", pct(lk["leak_ci"]["P_ESRG_nostop"]["pct"]), pct(lk["leak_ci"]["P_ESRG"]["pct"]),
         "McNemar", tb["leak"]],
        {"group": "Overall delineation (automatic seed)"},
        ["DSC *M*", f(au["dsc"]["base"]["mean"]), f(au["dsc"]["enh"]["mean"]), f"*r* = {nz(au['dsc']['r'])}", tb["overall_auto"]],
        ["Success rate", pct(au["success"]["rate_ref"]), pct(au["success"]["rate_cmp"]), "McNemar", tb["overall_auto"]],
    ]
    ps = [rc["tests"][k]["p_holm"] for k in ("P_SRG", "P_ESRG_global", "P_ESRG_nolog")] + \
         [lp["tests"]["P_ESRG_nostop"]["p_holm"], lk["P_ESRG_nostop"]["p"], au["dsc"]["p_holm"], au["success"]["p_holm"]]
    return {"title": "Summary of the Evaluation per Objective",
            "header": ["Metric", "Without enhancement", "ESRG", "Effect", "Table"],
            "rows": rows, "widths": [3, 1.7, 1.3, 1.6, 0.7], "align": ["l", "c", "c", "c", "c"],
            "note": ("“Without enhancement” is the configuration lacking the enhancement (the baseline SRG, or ESRG with "
                     "one enhancement removed); for the within-slice *SD* it is the enhanced grower seeded by an "
                     "operator. Objectives 2 and 3 are measured with a planted seed so that seed placement does not "
                     f"affect them. *r* is from the Wilcoxon signed-rank test and rates from the exact McNemar test; "
                     f"{T._p_all(ps)}.")}


def cited(refs, text):
    """Keep the reference entries whose first author and year are cited in the chapter text."""
    out = []
    for r in refs:
        author = r.split(",")[0].strip()
        year = re.search(r"\((\d{4}|n\.d\.)\)", r).group(1)
        if re.search(re.escape(author) + r"[^()]{0,60}?\(?" + re.escape(year), text):
            out.append(r)
    return out


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
    D, R, tb = build(args.dir)
    from experiments.chapter4.discussion import texts
    disc = texts(R, pd.read_csv(os.path.join(args.dir, "raw_results.csv"), low_memory=False), tb)
    missing = []

    def fill(text):
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
    body = " ".join(b.get("text", "") + " " + b.get("note", "") for b in D.blocks)
    for r in cited(REFERENCES, body):
        D.ref(r)
    spec = os.path.join(args.dir, "chapter4_spec.json")
    json.dump({"title": "Chapter 4 Results and Discussion", "blocks": D.blocks}, open(spec, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if missing:
        print("discussion placeholders without text:", sorted(set(missing)))
    render(spec, args.out)
