"""
build_chapter4_brats.py — Chapter 4 (Results and Discussion) of the ESRG thesis, BraTS 2023 test set.

Purpose : Report the single test-set evaluation (experiments/PROTOCOL_BRATS.md) with the tables,
          figures, and interpretation that Chapter 3 announces: overall performance (E1), Objective 1
          (seed hit rate, E2 seeding consistency), Objective 2 (E3, E4), Objective 3 (E5a, E5b, E6),
          and the verdicts of the pre-specified decision rules.
Function : every number is read from <dir>/results.json and <dir>/raw_results.csv (analyze_brats.py,
          evaluate.py), never typed; the figures come from figures_brats.py. The spec is rendered by
          render_docx.js into Chapter4_Results_and_Discussion.docx.
          CLI: python -P experiments/chapter4/build_chapter4_brats.py [--dir outputs/evaluation_brats/test]
                                                                      [--out Chapter4_Results_and_Discussion.docx]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from experiments.stats import fmt_p

MM = 240 / 512
TYPES = ["glioma", "meningioma"]
SCOPES = TYPES + ["pooled"]
SNAME = {"glioma": "Glioma", "meningioma": "Meningioma", "pooled": "Pooled"}
METRICS = ["dsc", "recall", "precision", "hd95_wc"]
MNAME = {"dsc": "DSC", "recall": "Recall", "precision": "Precision", "hd95_wc": "HD95"}
ARM = {"A_ESRG": "ESRG, automatic", "P_ESRG": "ESRG, planted", "P_SRG": "SRG, planted"}


# ─── formatting ──────────────────────────────────────────────────────────────
def f3(x): return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
def f2(x): return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"
def f1(x): return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"
def nz(x, k=2): return f"{x:.{k}f}".replace("0.", ".", 1) if x >= 0 else "−" + f"{-x:.{k}f}".replace("0.", ".", 1)
def sgn(x, k=3): return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x):.{k}f}"
def P(p): s = fmt_p(p); return f"*p* {s}" if s.startswith("<") else f"*p* = {s}"
def ptab(p): s = fmt_p(p); return s
def eff(r): a = abs(r); return "large" if a >= .5 else "medium" if a >= .3 else "small" if a >= .1 else "negligible"
def rp(t): return f"*r* = {nz(t['r'])}, {P(t['p_holm'])}"
def call_word(c): return {"better": "significantly better", "worse": "significantly worse", "no effect": "not significantly different"}[c]
def md(d, k="median"): return d.get(k)


def iqr(d, fmt=f3): return f"{fmt(d['median'])} [{fmt(d['q1'])}, {fmt(d['q3'])}]"
def msd(d, fmt=f3): return f"{fmt(d['mean'])} ({fmt(d['sd'])})"


class Spec:
    def __init__(self): self.B = []
    def h1(self, t): self.B.append({"t": "h1", "text": t})
    def h2(self, t): self.B.append({"t": "h2", "text": t})
    def h3(self, t): self.B.append({"t": "h3", "text": t})
    def h4(self, t): self.B.append({"t": "h4", "text": t})
    def p(self, t, **k): self.B.append({"t": "p", "text": t, **k})
    def ref(self, t): self.B.append({"t": "ref", "text": t})
    def table(self, num, title, header, rows, widths, note, align=None):
        self.B.append({"t": "table", "num": num, "title": title, "header": header, "rows": rows, "widths": widths,
                       "note": note, **({"align": align} if align else {})})
    def fig(self, num, img, title, note, w, h):
        self.B.append({"t": "figure", "num": num, "img": img, "title": title, "note": note, "w": w, "h": h})


def build(d, figdir):
    R = json.load(open(os.path.join(d, "results.json")))
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    figs = json.load(open(os.path.join(figdir, "figures.json")))
    n = R["n_slices"]
    nt = {t: int(df[df.tumor == t].file.nunique()) for t in TYPES}
    E1 = R["e1"]; E2 = R["e2"]; PA = R["paired"]; V = R["verdicts"]; D = R["descriptives"]
    post = lambda sc, m, k: E1[sc]["metrics"][m]["post"][k]
    dsc = lambda a, sc, k="mean": D[a][sc]["dsc"][k]
    test = lambda e, sc, m: next(x for x in PA[e][sc]["tests"] if x["metric"] == m)
    leak = df.groupby(["config", "tumor"]).leaked.mean().unstack() * 100
    leak_all = df.groupby("config").leaked.mean() * 100
    secs = df.groupby("config").seconds.median()
    errors = int(df.error.notna().sum()) if "error" in df else 0
    det = R["determinism"]
    det_ok = all(v["identical"] == v["n"] for v in det.values()) and len(det) == 4
    S = Spec()

    S.h1("Chapter Four")
    S.h1("RESULTS AND DISCUSSION")
    S.p("This chapter presents and interprets the results of the single evaluation of the test sample described in "
        "Section 3.1.1. Section 4.1 reports the overall performance of the proposed Enhanced Seeded Region Growing "
        "(ESRG) algorithm against the original Seeded Region Growing (SRG) algorithm; Section 4.2 reports the "
        "experiments of each objective; Section 4.3 describes the practical advantages of the ESRG; and Section 4.4 "
        "summarizes the findings against the decision rules fixed in "
        "Section 3.1.2. The configurations, metrics, and experiments are those defined in Section 3.1.2 and Table 3.2. "
        "Differences are oriented so that a positive value favors the enhanced configuration, and an effect is "
        "described as significant only when its Holm-adjusted *p* value is below .05 and |*r*| ≥ .10.")

    # ── 4.1 ──────────────────────────────────────────────────────────────────
    S.h2("4.1 General Results")
    S.p(f"All {n} sampled test patients ({nt['glioma']} glioma and {nt['meningioma']} meningioma, one slice each) "
        f"were processed by all twelve configurations, giving {len(df):,} segmentations"
        + (f", none of which ended in a processing error. " if errors == 0 else f", of which {errors} ended in a processing error and were excluded. ")
        + ("Both automatic configurations were run twice more on the same slices and produced bit-identical masks on "
           f"every slice in both repetitions, which confirms that the system is deterministic. " if det_ok else
           "The determinism check is reported in Section 4.2.1. ")
        + f"The median processing time per slice was {f1(secs['P_ESRG'])} s for the ESRG and {f1(secs['P_SRG'])} s for "
        "the SRG on the hardware of Table 3.4; these times are indicative only, because part of the run was performed "
        "on battery power with a reduced processor clock.")
    S.h3("4.1.1 Overall Performance")
    rows = []
    for sc in SCOPES:
        rows.append({"group": f"{SNAME[sc]} (n = {E1[sc]['metrics']['dsc']['n']})"})
        for a in ARM:
            dd = D[a][sc]
            rows.append([ARM[a], msd(dd["dsc"]), iqr(dd["dsc"]), msd(dd["recall"]), msd(dd["precision"]), iqr(dd["hd95_wc"], f1)])
    S.table("4.1", "Segmentation Performance of the Complete Configurations by Tumor Type",
            ["Configuration", "DSC, M (SD)", "DSC, Mdn [Q1, Q3]", "Recall, M (SD)", "Precision, M (SD)", "HD95 (px), Mdn [Q1, Q3]"],
            rows, [1.4, 1.0, 1.35, 1.0, 1.05, 1.5],
            f"DSC = Dice similarity coefficient; HD95 = 95th-percentile Hausdorff distance, in pixels (1 px = {MM:.3f} mm); "
            "M = mean; SD = standard deviation; Mdn = median; Q1, Q3 = first and third quartiles. Automatic = tumor seed "
            "selected by the system (A_ESRG); planted = seeds placed by the simulated operator at placement P1 (P_ESRG, "
            "P_SRG). Higher is better for DSC, recall, and precision; lower is better for HD95.",
            ["l", "c", "c", "c", "c", "c"])
    g, m_, pl = "glioma", "meningioma", "pooled"
    S.p(f"Table 4.1 shows that the original SRG with planted seeds obtained the highest mean DSC in both tumor types "
        f"({f3(dsc('P_SRG', g))} for glioma and {f3(dsc('P_SRG', m_))} for meningioma), followed by the ESRG with a "
        f"planted seed ({f3(dsc('P_ESRG', g))} and {f3(dsc('P_ESRG', m_))}) and by the ESRG with automatic seeding "
        f"({f3(dsc('A_ESRG', g))} and {f3(dsc('A_ESRG', m_))})." if dsc('P_SRG', g) >= dsc('P_ESRG', g) and dsc('P_SRG', m_) >= dsc('P_ESRG', m_) else
        f"Table 4.1 reports the mean DSC of the three configurations: {f3(dsc('A_ESRG', pl))} for the automatic ESRG, "
        f"{f3(dsc('P_ESRG', pl))} for the planted ESRG, and {f3(dsc('P_SRG', pl))} for the planted SRG (pooled).")
    pr = lambda a, sc: D[a][sc]["precision"]["mean"]; rc = lambda a, sc: D[a][sc]["recall"]["mean"]
    S.p(f"The two algorithms err in opposite directions. The planted ESRG has the higher precision (pooled mean "
        f"{f3(pr('P_ESRG', pl))} against {f3(pr('P_SRG', pl))}) but the lower recall ({f3(rc('P_ESRG', pl))} against "
        f"{f3(rc('P_SRG', pl))}): it rarely includes non-tumor tissue but stops before covering the whole tumor. The SRG, "
        "whose region is stopped only where a competing region arrives first, covers more of the tumor at the cost of "
        "more leakage. Both algorithms delineate meningioma, a well-circumscribed and homogeneously enhancing tumor, "
        "far better than glioma, whose infiltrative and heterogeneous core offers no sharp intensity boundary; on "
        f"meningioma the median DSC of every configuration exceeds 0.70, the level conventionally regarded as good "
        "agreement (Zijdenbos et al., 1994).")
    rows = []
    for sc in SCOPES:
        rows.append({"group": SNAME[sc]})
        for m in METRICS:
            e = E1[sc]["metrics"][m]; fr = e["friedman"]
            cell = lambda k: f"{sgn(e['post'][k]['median_paired_diff'], 1 if m == 'hd95_wc' else 3)} ({nz(e['post'][k]['r'])}{'' if e['post'][k]['call'] != 'no effect' else ', ns'})"
            rows.append([MNAME[m], f"{fr['chi2']:.2f}", ptab(fr["p"]), nz(fr["kendall_w"]),
                         cell("A_ESRG vs P_SRG"), cell("P_ESRG vs P_SRG"), cell("A_ESRG vs P_ESRG")])
    S.table("4.2", "Friedman Tests and Post Hoc Comparisons of the Complete Configurations (E1)",
            ["Metric", "χ^{2}(2)", "*p*", "*W*", "A_ESRG vs. P_SRG", "P_ESRG vs. P_SRG", "A_ESRG vs. P_ESRG"],
            rows, [1.0, 0.75, 0.65, 0.55, 1.35, 1.35, 1.35],
            "χ^{2}, *p*, and *W* are the Friedman statistic, its *p* value, and Kendall’s coefficient of concordance. Each "
            "post hoc cell gives the median paired difference (first configuration minus second; for HD95, second minus "
            "first, in pixels, so a positive value always favors the first configuration) and, in parentheses, the "
            "effect size *r* of the Wilcoxon signed-rank test. Post hoc *p* values were Holm-adjusted over the three "
            "pairs; ns = not significant (Holm *p* ≥ .05 or |*r*| < .10); all other post hoc differences are significant.",
            ["l", "c", "c", "c", "c", "c", "c"])
    a_s, p_s = post(pl, "dsc", "A_ESRG vs P_SRG"), post(pl, "dsc", "P_ESRG vs P_SRG")
    S.p(f"Table 4.2 confirms that the three configurations differ in every metric (pooled DSC: "
        f"χ^{{2}}(2) = {E1[pl]['metrics']['dsc']['friedman']['chi2']:.2f}, {P(E1[pl]['metrics']['dsc']['friedman']['p'])}, "
        f"*W* = {nz(E1[pl]['metrics']['dsc']['friedman']['kendall_w'])}). Against the planted SRG, the DSC of the "
        f"automatic ESRG is {call_word(a_s['call'])} ({rp(a_s)}), and that of the planted ESRG is "
        f"{call_word(p_s['call'])} ({rp(p_s)}). "
        + "By tumor type, " + "; ".join(f"for {sc}, the planted ESRG is {call_word(post(sc, 'dsc', 'P_ESRG vs P_SRG')['call'])} in DSC "
                    f"({rp(post(sc, 'dsc', 'P_ESRG vs P_SRG'))})" for sc in TYPES) + ". "
        f"In precision, the planted ESRG is {call_word(post(pl, 'precision', 'P_ESRG vs P_SRG')['call'])} than the SRG "
        f"({rp(post(pl, 'precision', 'P_ESRG vs P_SRG'))}), whereas in recall it is "
        f"{call_word(post(pl, 'recall', 'P_ESRG vs P_SRG')['call'])} ({rp(post(pl, 'recall', 'P_ESRG vs P_SRG'))}); "
        f"and its HD95 is {call_word(post(pl, 'hd95_wc', 'P_ESRG vs P_SRG')['call'])} "
        f"({rp(post(pl, 'hd95_wc', 'P_ESRG vs P_SRG'))}).")
    overall = V[pl]["overall"]
    S.p(("The overall improvement defined in Section 3.1.2 is therefore " + ("established" if overall else "not established")
         + ": " + ("the DSC of both ESRG configurations is significantly higher than that of the SRG."
                   if overall else
                   "under the conditions of this study, the complete ESRG does not delineate the tumor core more "
                   "accurately than the original SRG supplied with competing seeds. The gap is small for the planted "
                   f"ESRG (pooled median paired difference {sgn(p_s['median_paired_diff'])}) and is concentrated in recall, "
                   "which identifies undersegmentation, not leakage, as the remaining weakness of the proposed algorithm. "
                   "The larger gap of the automatic ESRG arises from the slices on which the automatic seed misses the "
                   "tumor (Section 4.2.1).")))
    S.fig("4.1", os.path.join(figdir, "figure_4_1.png"), "Distribution of the DSC of the Complete Configurations by Tumor Type",
          "Boxes span the first to third quartiles, the line marks the median, the diamond marks the mean, and the "
          "whiskers extend to 1.5 times the interquartile range; circles are individual slices beyond the whiskers. The "
          "dotted line marks a DSC of 0.70.", 1800, 840)
    zero = {t: (df[(df.config == "A_ESRG") & (df.tumor == t)].dsc.fillna(0) < 0.1).mean() * 100 for t in TYPES}
    S.p(f"Figure 4.1 shows why medians are reported alongside means. The distributions are skewed, with a tail of "
        f"near-complete failures that pulls the means below the medians. The tail is longest for the automatic ESRG on "
        f"glioma, where {zero['glioma']:.1f}% of slices have a DSC below 0.10 (against {zero['meningioma']:.1f}% for "
        "meningioma); such values are the signature of a seed placed outside the tumor rather than of an inaccurate "
        "boundary (Section 4.2.1). On meningioma, the boxes of all three configurations lie above the 0.70 line, and "
        "the SRG box is the narrowest and highest.")

    # ── 4.2 ──────────────────────────────────────────────────────────────────
    S.h2("4.2 Results per Objective")
    S.h3("4.2.1 Objective 1: Automatic Seed Selection")
    a = df[df.config == "A_ESRG"].copy(); a["hit"] = a.seed_hit.fillna(False).astype(bool)
    hitd = {sc: (a if sc == pl else a[a.tumor == sc]) for sc in SCOPES}
    rows = []
    for sc in SCOPES:
        s = E1[sc]["shr"]; h = hitd[sc]
        rows.append([SNAME[sc], f"{s['hits']} / {s['n']}", f"{s['shr']:.1f}%", f"[{s['ci'][0]:.1f}%, {s['ci'][1]:.1f}%]",
                     "Met" if s["ci"][0] >= 70 else "Not met", str(s["no_candidate"]),
                     f3(h[h.hit].dsc.mean()), f3(h[~h.hit].dsc.fillna(0).mean())])
    S.table("4.3", "Seed Hit Rate of Automatic Seed Selection and DSC of the Automatic ESRG by Seed Outcome",
            ["Tumor type", "Hits / n", "SHR", "95% CI", "LB ≥ 70%", "No candidate", "Mean DSC, hit", "Mean DSC, miss"],
            rows, [1.1, 0.85, 0.7, 1.25, 0.8, 0.8, 0.85, 0.85],
            "SHR = seed hit rate, the share of slices whose entire automatic seed core lies inside the ground-truth tumor "
            "core (Section 3.1.2); CI = Wilson score interval; LB = lower bound of the interval, compared with the 70% target "
            "of Objective 1. No candidate = slices on which the system reported “NO TUMOR CANDIDATE” (counted as misses, "
            "DSC = 0).", ["l", "c", "c", "c", "c", "c", "c", "c"])
    sh = {sc: E1[sc]["shr"] for sc in SCOPES}
    S.p(f"Table 4.3 shows that the automatic seed lies entirely inside the tumor core on {sh[pl]['hits']} of "
        f"{sh[pl]['n']} slices, a seed hit rate of {sh[pl]['shr']:.1f}% (95% CI [{sh[pl]['ci'][0]:.1f}%, "
        f"{sh[pl]['ci'][1]:.1f}%]). The rate depends strongly on the tumor type: {sh[m_]['shr']:.1f}% for meningioma, "
        f"whose lower confidence bound ({sh[m_]['ci'][0]:.1f}%) {'meets' if sh[m_]['ci'][0] >= 70 else 'falls short of'} "
        f"the 70% target, and {sh[g]['shr']:.1f}% for glioma, whose lower bound ({sh[g]['ci'][0]:.1f}%) "
        f"{'meets' if sh[g]['ci'][0] >= 70 else 'falls short of'} it. A meningioma typically enhances strongly and "
        "uniformly and is usually the brightest structure of the brain on a T1-CE slice, so the highest Otsu class isolates it. "
        "A glioma core is often a thin, irregular enhancing rim around necrosis, while vessels, the choroid plexus, and "
        "the dura can enhance as strongly as the tumor and may then be ranked above it. The seed outcome decides the "
        f"segmentation: when the seed hits the tumor, the automatic ESRG reaches a mean DSC of "
        f"{f3(hitd[pl][hitd[pl].hit].dsc.mean())}, at the level of the planted ESRG over all slices ({f3(dsc('P_ESRG', pl))}), whereas a miss "
        f"leaves a mean DSC of {f3(hitd[pl][~hitd[pl].hit].dsc.fillna(0).mean())}, because the region then grows "
        "inside the wrong structure.")
    S.fig("4.2", os.path.join(figdir, "figure_4_2.png"), "DSC of the Automatic ESRG When the Automatic Seed Hits or Misses the Tumor",
          "A hit means the entire automatic seed core lies inside the ground-truth tumor core. Slices without a candidate "
          "are counted as misses with DSC = 0. Box elements are as in Figure 4.1.", 1600, 800)
    S.p("Figure 4.2 makes this dependence visible: the two distributions overlap little. The weakness of the automatic "
        "pipeline therefore lies in localization, not in delineation, and it is confined mainly to glioma.")
    rows = []
    for exp, lab in (("E2a", "A_ESRG"), ("E2b", "A_SRG")):
        for sc in SCOPES:
            v = E2[exp][sc]
            rows.append([f"{exp} ({lab})", SNAME[sc], " / ".join(f3(v["dsc"][c]["median"]) for c in [v["auto"], "P_SRG", "P_SRG_2", "P_SRG_3"]),
                         f"{v['friedman']['chi2']:.2f}", ptab(v["friedman"]["p"]), nz(v["friedman"]["kendall_w"]),
                         "Yes" if v["auto_not_lower"] else "No"])
    S.table("4.4", "Consistency of the Result Across Seed Placements (E2)",
            ["Experiment", "Tumor type", "Median DSC: auto / P1 / P2 / P3", "χ^{2}(3)", "*p*", "*W*", "Auto not lower"],
            rows, [1.3, 1.1, 2.0, 0.8, 0.7, 0.6, 0.9],
            "Auto = the automatic configuration of the experiment (A_ESRG in E2a, A_SRG in E2b); P1, P2, P3 = the SRG with "
            "the planted seeds at the three placements (P_SRG, P_SRG_2, P_SRG_3). Auto not lower = no planted placement "
            "has a significantly higher DSC than the automatic configuration (Wilcoxon, Holm-adjusted over the three "
            "placements).", ["l", "l", "c", "c", "c", "c", "c"])
    w = df[df.config.isin(["P_SRG", "P_SRG_2", "P_SRG_3"])].pivot_table(index="file", columns="config", values="dsc")
    rng_ = (w.max(axis=1) - w.min(axis=1))
    S.p(f"Table 4.4 compares the automatic configurations with the three planted placements. Moving every planted "
        f"seed by 4 to 8 mm inside its own region changes the DSC of the SRG by at least 0.20 on "
        f"{(rng_ >= 0.2).mean()*100:.1f}% of slices, so where the operator clicks can decide the outcome. The automatic seed removes this source of variability "
        + ("entirely, as the repeated runs confirm (Section 4.1). " if det_ok else ". ")
        + f"However, consistency is not the same as accuracy. In E2a the median DSC of the automatic ESRG "
          f"({f3(E2['E2a'][pl]['dsc']['A_ESRG']['median'])}) is "
          f"{'not significantly lower than' if E2['E2a'][pl]['auto_not_lower'] else 'significantly lower than'} that of "
          f"the planted placements, and in E2b the SRG started from the same automatic tumor seed reaches a median of "
          f"{f3(E2['E2b'][pl]['dsc']['A_SRG']['median'])}, which is "
          f"{'not significantly lower than' if E2['E2b'][pl]['auto_not_lower'] else 'also significantly lower than'} that of the planted placements. "
        + f"Because A_ESRG and A_SRG start from the same automatic tumor seed, the difference between them (median "
          f"{f3(E2['E2a'][pl]['dsc']['A_ESRG']['median'])} against {f3(E2['E2b'][pl]['dsc']['A_SRG']['median'])}) reflects "
          "the growth rule, whereas the shortfall of both relative to the planted placements reflects the seed hit rate.")
    o1 = V[pl]["objective1"]
    S.p(f"With respect to Objective 1, the three criteria of Section 3.1.2 give the following result. The seed hit rate "
        f"criterion is {'met' if sh[m_]['ci'][0] >= 70 else 'not met'} for meningioma and "
        f"{'met' if sh[g]['ci'][0] >= 70 else 'not met'} for glioma "
        f"({'met' if o1['a_shr_lb_ge_70'] else 'not met'} pooled); the consistency criterion is "
        f"{'met' if o1['b_consistency'] else 'not met'}; and the criterion that the automatic DSC is not lower than any "
        f"planted placement is {'met' if o1['c_auto_not_lower'] else 'not met'} pooled. Objective 1 is therefore "
        + ("met." if all(o1.values()) else
           "met only in part: automatic seed selection makes the result reproducible and is reliable for meningioma, "
           "but it does not yet localize gliomas reliably enough to replace a careful operator."))

    # Objective 2
    S.h3("4.2.2 Objective 2: Minimization of Undersegmentation")
    def pair_rows(exps, metrics_by):
        rows = []
        for e, lab in exps:
            base, enh = PA[e][pl]["baseline"], PA[e][pl]["enhanced"]
            rows.append({"group": f"{e}: {lab} ({base} → {enh})"})
            for sc in SCOPES:
                for t in PA[e][sc]["tests"]:
                    k = 1 if t["metric"] == "hd95_wc" else 3
                    rows.append([SNAME[sc], MNAME[t["metric"]] + (" (no-harm)" if t["role"] == "guard" else ""),
                                 iqr(t["base"], f1 if k == 1 else f3), iqr(t["enh"], f1 if k == 1 else f3),
                                 sgn(t["median_paired_diff"], k), f"{t['n_pos']} / {t['n_neg']}", nz(t["r"]), ptab(t["p_holm"]),
                                 {"better": "Better", "worse": "Worse", "no effect": "ns"}[t["call"]]])
        return rows
    hdr = ["Tumor type", "Metric", "Baseline, Mdn [Q1, Q3]", "Enhanced, Mdn [Q1, Q3]", "Mdn diff.", "+ / −", "*r*", "*p*_{Holm}", "Result"]
    wid = [0.95, 1.0, 1.35, 1.35, 0.7, 0.75, 0.5, 0.6, 0.6]
    note2 = ("Baseline → enhanced as defined in Table 3.2. Mdn diff. = median paired difference, oriented so that a positive "
             "value favors the enhanced configuration (for HD95, baseline minus enhanced, in pixels); + / − = number of "
             "slices on which the enhanced configuration was better / worse; *r* = effect size of the Wilcoxon signed-rank "
             "test; *p*_{Holm} = Holm-adjusted over the metrics of the experiment. ns = not significant (Holm *p* ≥ .05 or "
             "|*r*| < .10). No-harm = the DSC check of Section 3.1.2.")
    S.table("4.5", "Effect of the Local Log-Domain Difference Measure on Recall and DSC (E3 and E4)", hdr,
            pair_rows([("E3", "LLDM added to SRG"), ("E4", "LLDM removed from ESRG")], None), wid, note2,
            ["l", "l", "c", "c", "c", "c", "c", "c", "c"])
    e3r, e3d, e4r, e4d = test("E3", pl, "recall"), test("E3", pl, "dsc"), test("E4", pl, "recall"), test("E4", pl, "dsc")
    S.p(f"Table 4.5 answers two questions. E3 asks whether the local log-domain difference measure (LLDM) is "
        f"sufficient to reduce undersegmentation when it is added to the original SRG. It raises recall "
        f"({call_word(e3r['call'])}, median {f3(e3r['base']['median'])} → {f3(e3r['enh']['median'])}, {rp(e3r)}), but the "
        f"DSC collapses (median {f3(e3d['base']['median'])} → {f3(e3d['enh']['median'])}, {rp(e3d)}); the region "
        f"covered more than 40% of the brain on {leak_all.get('P_SRG_L', float('nan')):.1f}% of slices, against "
        f"{leak_all.get('P_SRG', float('nan')):.1f}% for the plain SRG. The higher recall is therefore the by-product of "
        "flooding, not of a better boundary. The reason lies in the competition of the original algorithm. The tumor "
        "region of the SRG competes with the healthy-tissue and background regions for every pixel, and the global mean "
        "of the tumor region is what makes healthy tissue look different from the tumor. A local reference follows the "
        "intensity of the most recently absorbed neighbors, so once the tumor region reaches its boundary, the healthy "
        "tissue next to it looks similar and is absorbed before the healthy-tissue region arrives. Without a stopping "
        "rule, a local measure removes the only brake that the SRG has.")
    S.p(f"E4 asks whether the measure is necessary within the ESRG, where the adaptive stopping criterion bounds the "
        f"region. With the measure, the pooled recall of the ESRG is {call_word(e4r['call'])} than without it "
        f"(median {f3(e4r['base']['median'])} → {f3(e4r['enh']['median'])}, {rp(e4r)}), and its DSC is "
        f"{call_word(e4d['call'])} ({rp(e4d)}). By tumor type, the recall with the measure is "
        + " and ".join(f"{call_word(test('E4', sc, 'recall')['call'])} for {sc} ({rp(test('E4', sc, 'recall'))})" for sc in TYPES)
        + ". Inside the ESRG, the comparison with the adjacent tumor tissue lets the region follow the gradual intensity "
          "changes of a heterogeneous tumor, while the stopping bounds keep it from following those changes into the "
          "surrounding tissue.")
    S.p("Objective 2 is " + ("met." if V[pl]["objective2"] else
        "therefore not met under the strict rule of Section 3.1.2, which requires a significant recall gain without loss of DSC in both "
        "E3 and E4. The results support a narrower conclusion: the local log-domain measure is "
        + ("necessary for the recall and DSC of the ESRG on meningioma and pooled, although not on glioma" if e4r["call"] == "better" and e4d["call"] != "worse" else "not clearly beneficial even within the ESRG")
        + ", but it is not sufficient on its own and is harmful in an algorithm that lacks a stopping criterion."))

    # Objective 3
    S.h3("4.2.3 Objective 3: Decrease of Boundary Leakage")
    S.table("4.6", "Effect of the Adaptive Stopping Criterion on Precision, HD95, and DSC (E5a, E5b, and E6)", hdr,
            pair_rows([("E5a", "ASC with one seed vs. SRG with three seeds"), ("E5b", "ASC added to SRG with one seed"),
                       ("E6", "ASC removed from ESRG")], None), wid,
            note2 + " The baseline of E5b, the SRG with a single seed and no stopping rule, absorbs the whole image by design.",
            ["l", "l", "c", "c", "c", "c", "c", "c", "c"])
    e6p, e6h, e6d = test("E6", pl, "precision"), test("E6", pl, "hd95_wc"), test("E6", pl, "dsc")
    e5p, e5h, e5d = test("E5a", pl, "precision"), test("E5a", pl, "hd95_wc"), test("E5a", pl, "dsc")
    e5bp = test("E5b", pl, "precision")
    S.p(f"Table 4.6 shows the strongest effects of the study. E6 removes the adaptive stopping criterion (ASC) from the "
        f"ESRG while keeping its competing regions. Without it, the median precision is {f3(e6p['base']['median'])} and "
        f"the region exceeds 40% of the brain on {leak_all.get('P_ESRG_C_noS', float('nan')):.1f}% of slices; with it, "
        f"the median precision is {f3(e6p['enh']['median'])} ({rp(e6p)}), the median HD95 falls from "
        f"{f1(e6h['base']['median'])} to {f1(e6h['enh']['median'])} pixels ({rp(e6h)}), and the median DSC rises from "
        f"{f3(e6d['base']['median'])} to {f3(e6d['enh']['median'])} ({rp(e6d)}). Because the local measure removes the "
        "brake of the global mean (Section 4.2.2), the stopping criterion is what keeps the ESRG inside the tumor; it is "
        "necessary within the proposed algorithm. E5b shows the same mechanism in the original algorithm: a single SRG "
        f"region absorbs the entire image (median precision {f3(e5bp['base']['median'])}), whereas the same region under "
        f"the ASC reaches a median precision of {f3(e5bp['enh']['median'])} ({rp(e5bp)}).")
    S.p(f"E5a is the demanding comparison: it asks whether the ASC, from the tumor seed alone, can replace the two "
        f"competing seeds that the original algorithm needs. The SRG with the ASC and one seed has "
        f"{call_word(e5p['call']).replace('significantly ', 'significantly ')} precision than the SRG with three seeds "
        f"(median {f3(e5p['base']['median'])} → {f3(e5p['enh']['median'])}, {rp(e5p)}), so it leaks less. Its HD95 is "
        f"{call_word(e5h['call'])} ({rp(e5h)}) and its DSC is {call_word(e5d['call'])} (median "
        f"{f3(e5d['base']['median'])} → {f3(e5d['enh']['median'])}, {rp(e5d)}). The stopping bounds thus keep the region "
        "inside the tumor more strictly than the competing regions do, but they also stop it earlier, before it has "
        "covered the darker or more heterogeneous parts of the tumor; the boundary errors move from leakage to "
        "undersegmentation, which HD95 and DSC penalize.")
    S.p("Objective 3 is " + ("met." if V[pl]["objective3"] else
        "therefore not met under the strict rule of Section 3.1.2, because E5a does not show a lower HD95 and an "
        "unharmed DSC. The criterion does, however, decrease boundary leakage in every comparison, as measured by "
        "precision, and its removal from the ESRG causes one of the largest losses of accuracy observed in the study."))
    S.fig("4.3", os.path.join(figdir, "figure_4_3.png"), "Segmentations of Two Test Slices by Six Configurations",
          "Red fill and outline = predicted mask; yellow outline = ground-truth tumor core where it is not covered by the "
          "prediction. In each row the slice is the test slice whose SRG (planted) DSC is closest to the median of its "
          "tumor type, so the slices are typical for the baseline, not selected for the ESRG. LLDM = local log-domain "
          "difference measure; ASC = adaptive stopping criterion.", 3080, 1060)
    ex = {t: df[df.file == fname].set_index("config").dsc for t, fname in zip(TYPES, figs["examples"])}
    S.p(f"Figure 4.3 illustrates these mechanisms on two test slices that are typical for the baseline. Adding the "
        f"local measure to the SRG and removing the stopping criterion from the ESRG flood the brain in both rows "
        f"(DSC {f2(ex['glioma']['P_SRG_L'])} and {f2(ex['meningioma']['P_SRG_L'])} for the SRG with the local measure). "
        f"The SRG with the stopping criterion and a single seed stays inside the tumor but stops early (DSC "
        f"{f2(ex['glioma']['P_SRG_solo_S'])} and {f2(ex['meningioma']['P_SRG_solo_S'])}). The glioma row also shows how "
        f"strongly the ESRG depends on its seed, whose statistics set the stopping bounds. Started from the planted seed "
        f"at the deepest point of the core, the ESRG covers only part of the core (DSC "
        f"{f2(ex['glioma']['P_ESRG'])}); started from the automatic seed, it delineates the core almost completely (DSC {f2(ex['glioma']['A_ESRG'])}, against {f2(ex['glioma']['P_SRG'])} "
        f"for the SRG). On the meningioma, the ESRG (DSC {f2(ex['meningioma']['P_ESRG'])}) misses a thin margin of the "
        f"tumor that the SRG (DSC {f2(ex['meningioma']['P_SRG'])}) includes, which is the recall deficit of Table 4.1 in "
        "visual form.")

    # ── 4.3 Practical advantages ─────────────────────────────────────────────
    S1 = R["secondary"]["S1"]
    s1 = lambda sc, met: next(x for x in S1[sc]["tests"] if x["metric"] == met)
    w3 = df[df.config.isin(["P_SRG", "P_SRG_2", "P_SRG_3"])].pivot_table(index="file", columns="config", values="dsc")
    vary = ((w3.max(axis=1) - w3.min(axis=1)) >= 0.2).mean() * 100
    ident = sum(v["identical"] for k, v in det.items() if k.startswith("A_ESRG"))
    ntot = sum(v["n"] for k, v in det.items() if k.startswith("A_ESRG"))
    pe = post(pl, "precision", "P_ESRG vs P_SRG")
    S.h2("4.3 Practical Advantages of the ESRG")
    S.p("The decision rules of Section 3.1.2 compare the algorithms under the conditions most favorable to the "
        "original SRG: three correctly placed seeds per slice. The test data also show where the ESRG offers a "
        "practical advantage. These findings complement the primary results above and do not replace them. "
        "Table 4.7 lists them; the comparison S1 was defined after the test run (Section 3.1.2) and is therefore "
        "descriptive.")
    S.table("4.7", "Practical Advantages of the ESRG Observed on the Test Sample",
            ["Aspect", "ESRG", "SRG", "Evidence"],
            [["Seeds the operator must place", "One tumor click, or none (automatic)", "Tumor, healthy-tissue, and background seeds", "Table 3.2"],
             ["Same single tumor click (S1), median DSC", f3(s1(pl, "dsc")["enh"]["median"]), f3(s1(pl, "dsc")["base"]["median"]),
              f"Wilcoxon {rp(s1(pl, 'dsc'))}; secondary"],
             ["Same single tumor click (S1), median precision", f3(s1(pl, "precision")["enh"]["median"]), f3(s1(pl, "precision")["base"]["median"]),
              f"Wilcoxon {rp(s1(pl, 'precision'))}; secondary"],
             ["Leakage with all seeds placed, mean precision", f3(pr("P_ESRG", pl)), f3(pr("P_SRG", pl)), f"E1, {rp(pe)}"],
             ["Reproducibility", f"Automatic masks identical on {ident:,} of {ntot:,} repeated runs",
              f"DSC range ≥ 0.20 across the three placements on {vary:.1f}% of slices", "Sections 4.1 and 4.2.1"],
             ["Meningioma without user input", f"Median DSC {f3(D['A_ESRG'][m_]['dsc']['median'])}; SHR {sh[m_]['shr']:.1f}%",
              "Requires three seeds", "Tables 4.1 and 4.3"],
             ["Median processing time per slice", f"{f1(secs['P_ESRG'])} s", f"{f1(secs['P_SRG'])} s", "Indicative only (Section 4.1)"]],
            [1.9, 1.6, 1.7, 1.6],
            "S1 compares P_ESRG with P_SRG_solo, in which both algorithms receive only the planted tumor seed. S1 was added "
            "after the test run and does not enter the decision rules. The original SRG is not designed to run with a single "
            "seed, so S1 measures the consequence of the input each algorithm requires, not the quality of its boundary "
            "rule. SHR = seed hit rate.",
            ["l", "l", "l", "l"])
    S.p(f"Table 4.7 shows that the main practical difference between the algorithms is the input they require. The "
        f"original SRG reaches its accuracy only when the operator also places correct healthy-tissue and background "
        f"seeds; given the same single tumor click as the ESRG, its region absorbs the whole image (median DSC "
        f"{f3(s1(pl, 'dsc')['base']['median'])}), whereas the ESRG stops at the tumor (median DSC "
        f"{f3(s1(pl, 'dsc')['enh']['median'])}, {rp(s1(pl, 'dsc'))}). This is a direct consequence of the adaptive "
        "stopping criterion and reflects the purpose of Objective 3: the ESRG controls leakage without competing "
        f"seeds. Even when the SRG receives all three seeds, the ESRG leaks less (mean precision {f3(pr('P_ESRG', pl))} "
        f"against {f3(pr('P_SRG', pl))}, {rp(pe)}). The ESRG with automatic seeding also removes the operator "
        f"entirely: its result was identical in every repeated run, it needs no click at all, and on meningioma it "
        f"reaches a median DSC of {f3(D['A_ESRG'][m_]['dsc']['median'])}. These advantages come at the cost "
        f"reported in Section 4.1: lower recall, and therefore a lower DSC than the SRG with three correct seeds.")

    # ── 4.4 Summary ──────────────────────────────────────────────────────────
    S.h2("4.4 Summary of Findings")
    yn = lambda b: "Met" if b else "Not met"
    sup = {
        "overall": f"No. Precision higher (E1, *r* = {nz(pe['r'])}), but recall and DSC lower",
        "o1a": f"Meningioma only (lower bound {sh[m_]['ci'][0]:.1f}%)",
        "o1b": f"Yes ({ident:,} of {ntot:,} repeated masks identical)",
        "o1c": f"No overall; when the seed hits, mean DSC {f3(hitd[pl][hitd[pl].hit].dsc.mean())}",
        "o2": "Within the ESRG only: recall higher in E4 (pooled and meningioma); E3 floods",
        "o3": "For leakage: precision higher in E5a, E5b, and E6; HD95 and DSC not improved in E5a",
    }
    rows = [["Overall improvement", "A_ESRG and P_ESRG DSC significantly higher than P_SRG (E1)"] + [yn(V[sc]["overall"]) for sc in SCOPES] + [sup["overall"]],
            ["Objective 1", "SHR lower bound ≥ 70% (E1)"] + [yn(V[sc]["objective1"]["a_shr_lb_ge_70"]) for sc in SCOPES] + [sup["o1a"]],
            ["", "Automatic result invariant; planted results vary (E2)"] + [yn(V[sc]["objective1"]["b_consistency"]) for sc in SCOPES] + [sup["o1b"]],
            ["", "Automatic DSC not lower than any placement (E2a)"] + [yn(V[sc]["objective1"]["c_auto_not_lower"]) for sc in SCOPES] + [sup["o1c"]],
            ["Objective 2", "Recall higher, DSC not lower in E3 and E4"] + [yn(V[sc]["objective2"]) for sc in SCOPES] + [sup["o2"]],
            ["Objective 3", "Precision higher, HD95 lower, DSC not lower in E5a and E6"] + [yn(V[sc]["objective3"]) for sc in SCOPES] + [sup["o3"]]]
    S.table("4.8", "Results of the Pre-Specified Decision Rules and the Partial Support Observed",
            ["Criterion", "Decision rule (Section 3.1.2)", "Glioma", "Meningioma", "Pooled", "What the data support"],
            rows, [1.15, 2.0, 0.7, 0.85, 0.7, 2.0],
            "Glioma, Meningioma, and Pooled give the verdict of the pre-specified rule; a difference counts only if its "
            "Holm-adjusted *p* value is below .05 and |*r*| ≥ .10. The last column describes the part of the hypothesis "
            "that the results support; it does not change the verdicts. SHR = seed hit rate.",
            ["l", "l", "c", "c", "c", "l"])
    S.p("Table 4.8 applies the decision rules that were fixed before the test set was evaluated. Read strictly, the "
        "rules do not establish an overall improvement and do not mark any objective as fully met; this result is "
        "reported as it was obtained. The last column shows which parts of each hypothesis the data do support: the "
        "automatic seed makes the result reproducible and localizes meningiomas reliably, the local log-domain "
        "measure improves recall within the ESRG, and the adaptive stopping criterion decreases boundary leakage in "
        "every comparison. The findings are summarized below.")
    S.p("First, the enhancements act as a coupled system rather than as independent improvements. The local log-domain "
        "measure lets the region follow the tumor’s internal intensity variations, but it also lets the region follow "
        "smooth transitions into healthy tissue; the adaptive stopping criterion bounds that growth. Neither method "
        "improves the DSC of the original algorithm on its own (E3, E5a), whereas removing either from the ESRG "
        "lowers its accuracy (E4, E6). The proposed algorithm should therefore be judged as a whole, against the "
        "original algorithm, which is the comparison of E1.")
    S.p(f"Second, in that comparison the ESRG trades recall for precision. It leaks less than the SRG "
        f"(pooled mean precision {f3(pr('P_ESRG', pl))} against {f3(pr('P_SRG', pl))}) but misses more of the tumor "
        f"(recall {f3(rc('P_ESRG', pl))} against {f3(rc('P_SRG', pl))}), and in DSC it "
        f"{'does not surpass' if not overall else 'surpasses'} the SRG supplied with a healthy-tissue and a background "
        "seed. The original algorithm with competing seeds is a stronger baseline than its reputation for leakage "
        "suggests: competition stops leakage effectively when the operator supplies seeds for the surrounding "
        "tissues. The ESRG achieves higher precision from a single tumor seed, without any competing seed, which "
        "is its practical advantage, but its stopping bounds are conservative on heterogeneous tumors.")
    S.p(f"Third, the class dependence is decisive. On meningioma, both planted algorithms reach a median DSC of "
        f"{f3(D['P_ESRG'][m_]['dsc']['median'])} (ESRG) and {f3(D['P_SRG'][m_]['dsc']['median'])} (SRG), and the "
        f"automatic seed is placed correctly on {sh[m_]['shr']:.0f}% of slices, so the fully automatic ESRG yields a "
        f"good segmentation without any user input (median DSC {f3(D['A_ESRG'][m_]['dsc']['median'])}). On glioma, "
        "the infiltrative and heterogeneous enhancement limits every intensity-based method: the automatic seed misses "
        "often, and both growth rules leave a substantial part of the core unsegmented or include adjacent enhancing "
        "structures. Region growing on a single intensity channel thus "
        "performs best on well-circumscribed, homogeneously enhancing lesions.")
    S.p("Finally, the study design itself contributes to the interpretation. The SRG was given the same seeds, the same "
        "brain mask, and, for every applied method, the same parameter values as the ESRG, and it was implemented as "
        "published, including 8-connectivity. The planted seeds were placed by a simulated careful operator with access "
        "to the ground truth: every planted seed, including the healthy-tissue and background seeds on which the SRG "
        "depends, was placed correctly. The planted SRG therefore represents the original algorithm under favorable "
        "conditions, and a real operator, whose clicks vary (Section 4.2.1), may obtain lower scores with it.")

    S.h2("References Added in Chapter 4")
    S.p("The following works are cited in this chapter; the remaining citations are listed in Chapter 3.", noindent=True)
    if False: S.ref("Biratu, E. S., Schwenker, F., Debelee, T. G., Kebede, S. R., Negera, W. G., & Molla, H. T. (2021). Enhanced region "
          "growing for brain tumor MR image segmentation. *Journal of Imaging, 7*(2), Article 22. https://doi.org/10.3390/jimaging7020022")
    S.ref("Zijdenbos, A. P., Dawant, B. M., Margolin, R. A., & Palmer, A. C. (1994). Morphometric analysis of white matter "
          "lesions in MR images: Method and validation. *IEEE Transactions on Medical Imaging, 13*(4), 716–724. "
          "https://doi.org/10.1109/42.363096")
    return {"title": "Chapter 4: Results and Discussion", "author": "Karylle Vinces J. Aliswag & Juana Mari A. Del Rosario", "blocks": S.B}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "evaluation_brats", "test"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "outputs", "chapter4_brats"))
    ap.add_argument("--out", default=os.path.join(ROOT, "Chapter4_Results_and_Discussion.docx"))
    ap.add_argument("--no-figures", action="store_true", help="reuse the figures already in --figdir")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    if not a.no_figures:
        subprocess.run([sys.executable, "-P", os.path.join(HERE, "figures_brats.py"), a.dir, a.figdir], check=True)
    spec = build(a.dir, a.figdir)
    sp = os.path.join(a.figdir, "chapter4_spec.json")
    json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    subprocess.run(["node", os.path.join(HERE, "render_docx.js"), sp, a.out], check=True)
    print("blocks:", len(spec["blocks"]))
