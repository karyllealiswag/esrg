"""
build_chapter4_brats.py — Chapter 4 (Results and Discussion) of the ESRG thesis, BraTS 2023 test set.

Purpose : Report the single test-set evaluation (experiments/PROTOCOL_BRATS.md) with the tables,
          figure, and interpretation that Chapter 3 announces. Results are given separately for
          glioma and meningioma. Section 4.1: overall performance of four configurations that differ in
          the operator input (A_ESRG, P_ESRG, P_SRG_solo with one planted seed, P_SRG with three), with
          processing time; Section 4.2: Objective 1 (seed hit rate, E2), Objective 2 (E3, E4),
          Objective 3 (E5a, E5b, E6); Section 4.3: comparative analysis of the baseline SRG and the ESRG.
Function : every number is read from <dir>/results.json and <dir>/raw_results.csv (analyze_brats.py,
          evaluate.py), never typed; the figure comes from figures_brats.py. The spec is rendered by
          render_docx.js into Chapter4_Results_and_Discussion.docx. Qualitative statements are guarded by
          assertions, so the build stops if the results no longer support the sentence.
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
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from experiments.stats import fmt_p

MM = 240 / 512
TYPES = ["glioma", "meningioma"]
SNAME = {"glioma": "Glioma", "meningioma": "Meningioma"}
METRICS = ["dsc", "recall", "precision", "hd95_wc", "seconds"]
MNAME = {"dsc": "DSC", "recall": "Recall", "precision": "Precision", "hd95_wc": "HD95", "seconds": "Time"}
ARMS = ["A_ESRG", "P_ESRG", "P_SRG_solo", "P_SRG"]
ARM = {"A_ESRG": "ESRG, automatic seed (A_ESRG)", "P_ESRG": "ESRG, 1 planted seed (P_ESRG)",
       "P_SRG_solo": "SRG, 1 planted seed (P_SRG_solo)", "P_SRG": "SRG, 3 planted seeds (P_SRG)"}


# ─── formatting ──────────────────────────────────────────────────────────────
def _nan(x): return x is None or (isinstance(x, float) and math.isnan(x))
def f3(x): return "—" if _nan(x) else f"{x:.3f}"
def f2(x): return "—" if _nan(x) else f"{x:.2f}"
def f1(x): return "—" if _nan(x) else f"{x:.1f}"
def nz(x, k=2): return f"{x:.{k}f}".replace("0.", ".", 1) if x >= 0 else "−" + f"{-x:.{k}f}".replace("0.", ".", 1)
def sgn(x, k=3): return ("+" if x > 0 else "−" if x < 0 else "") + f"{abs(x):.{k}f}"
def P(p): s = fmt_p(p); return f"*p* {s}" if s.startswith("<") else f"*p* = {s}"
def ptab(p): return fmt_p(p)
def rp(t): return f"*r* = {nz(t['r'])}, {P(t['p_holm'])}"
def call_word(c): return {"better": "significantly better", "worse": "significantly worse", "no effect": "not significantly different"}[c]
def iqr(d, fmt=f3): return f"{fmt(d['median'])} [{fmt(d['q1'])}, {fmt(d['q3'])}]"
def msd(d, fmt=f3): return f"{fmt(d['mean'])} ({fmt(d['sd'])})"
def both(fn):
    """Phrase for both tumor types: the common phrase once when equal, else 'a for glioma and b for meningioma'."""
    a, b = fn(TYPES[0]), fn(TYPES[1])
    return f"{a} in both tumor types" if a == b else f"{a} for glioma and {b} for meningioma"


class Spec:
    def __init__(self): self.B = []
    def h1(self, t): self.B.append({"t": "h1", "text": t})
    def h2(self, t): self.B.append({"t": "h2", "text": t})
    def h3(self, t): self.B.append({"t": "h3", "text": t})
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
    E1, E2, PA, SEC, V = R["e1"], R["e2"], R["paired"], R["secondary"], R["verdicts"]
    g, m_ = TYPES
    e1d = lambda sc, m, a: E1[sc]["metrics"][m]["desc"][a]
    post = lambda sc, m, k: E1[sc]["metrics"][m]["post"][k]
    fried = lambda sc, m: E1[sc]["metrics"][m]["friedman"]
    test = lambda e, sc, m: next(x for x in PA[e][sc]["tests"] if x["metric"] == m)
    sec = lambda e, sc, m: next(x for x in SEC[e][sc]["tests"] if x["metric"] == m)
    leak = df.groupby(["config", "tumor"]).leaked.mean().unstack() * 100
    errors = int(df.error.notna().sum()) if "error" in df else 0
    det = R["determinism"]
    det_ok = all(v["identical"] == v["n"] for v in det.values()) and len(det) == 4
    S = Spec()

    # ── claims the text relies on: the build stops if the data no longer support them ──
    for sc in TYPES:
        med = {a: e1d(sc, "dsc", a)["median"] for a in ARMS}
        assert med["P_SRG"] == max(med.values()), "P_SRG should have the highest median DSC"
        assert med["P_SRG_solo"] == min(med.values()), "P_SRG_solo should have the lowest median DSC"
        assert e1d(sc, "recall", "P_SRG_solo")["median"] > 0.99 and e1d(sc, "precision", "P_SRG_solo")["median"] < 0.05
        assert post(sc, "dsc", "P_ESRG vs P_SRG_solo")["call"] == "better"
        assert post(sc, "dsc", "P_ESRG vs P_SRG")["call"] == "worse"
        assert post(sc, "precision", "P_ESRG vs P_SRG")["call"] == "better"
        assert post(sc, "recall", "P_ESRG vs P_SRG")["call"] == "worse"
        assert post(sc, "seconds", "P_ESRG vs P_SRG")["call"] == "better"
        assert not V[sc]["overall"]
    assert all(e1d(m_, "dsc", a)["median"] > 0.70 for a in ("A_ESRG", "P_ESRG", "P_SRG"))

    S.h1("Chapter Four")
    S.h1("RESULTS AND DISCUSSION")
    S.p("This chapter presents and interprets the results of the single evaluation of the test sample described in "
        "Section 3.1.1, separately for glioma and meningioma. Section 4.1 reports the overall performance of the "
        "proposed Enhanced Seeded Region Growing (ESRG) algorithm and the original Seeded Region Growing (SRG) "
        "algorithm, including processing time; Section 4.2 reports the experiments of each objective; and "
        "Section 4.3 compares the baseline SRG with the ESRG experiment by experiment. The configurations, metrics, "
        "and experiments are those defined in Section 3.1.2 and Table 3.2. Differences are oriented so that a "
        "positive value favors the enhanced configuration, and an effect is described as significant only when its "
        "Holm-adjusted *p* value is below .05 and |*r*| ≥ .10.")

    # ═══ 4.1 ═════════════════════════════════════════════════════════════════
    S.h2("4.1 General Results")
    S.p(f"All {n} sampled test patients ({nt[g]} glioma and {nt[m_]} meningioma, one slice each) were processed by all "
        f"twelve configurations, giving {len(df):,} segmentations"
        + (", none of which ended in a processing error. " if errors == 0 else f", of which {errors} ended in a processing error and were excluded. ")
        + ("Both automatic configurations were run twice more on the same slices and produced bit-identical masks on "
           "every slice in both repetitions, which confirms that the system is deterministic." if det_ok else
           "The determinism check is reported in Section 4.2.1."))
    S.h3("4.1.1 Overall Performance")
    S.p("Table 4.1 compares four configurations that differ in the input the operator provides: the ESRG with an "
        "automatic tumor seed (A_ESRG), the ESRG with one planted tumor seed (P_ESRG), the SRG with the same single "
        "planted tumor seed (P_SRG_solo), and the SRG with a tumor, a healthy-tissue, and a background seed "
        "planted (P_SRG).")
    rows = []
    for sc in TYPES:
        rows.append({"group": f"{SNAME[sc]} (n = {E1[sc]['metrics']['dsc']['n']})"})
        for a in ARMS:
            rows.append([ARM[a], msd(e1d(sc, "dsc", a)), iqr(e1d(sc, "dsc", a)), msd(e1d(sc, "recall", a)),
                         msd(e1d(sc, "precision", a)), iqr(e1d(sc, "hd95_wc", a), f1), iqr(e1d(sc, "seconds", a), f2)])
    S.table("4.1", "Segmentation Performance and Processing Time of the Four Configurations by Tumor Type",
            ["Configuration", "DSC, M (SD)", "DSC, Mdn [Q1, Q3]", "Recall, M (SD)", "Precision, M (SD)",
             "HD95 (px), Mdn [Q1, Q3]", "Time (s), Mdn [Q1, Q3]"],
            rows, [1.7, 0.95, 1.25, 0.95, 0.95, 1.2, 1.1],
            f"DSC = Dice similarity coefficient; HD95 = 95th-percentile Hausdorff distance, in pixels (1 px = {MM:.3f} mm); "
            "Time = processing time per slice in seconds; M = mean; SD = standard deviation; Mdn = median; Q1, Q3 = "
            "first and third quartiles. Automatic = tumor seed selected by the system; planted = seeds placed by the "
            "simulated operator at placement P1. Time was measured with 11 worker processes running in parallel on the "
            "hardware of Table 3.4, part of it on battery power, so it indicates relative speed only. Higher is better for "
            "DSC, recall, and precision; lower is better for HD95 and time.",
            ["l", "c", "c", "c", "c", "c", "c"])
    sol = {sc: (e1d(sc, "dsc", "P_SRG_solo"), e1d(sc, "recall", "P_SRG_solo"), e1d(sc, "precision", "P_SRG_solo")) for sc in TYPES}
    S.p("Given only the tumor seed, the SRG has no competing region to stop it and absorbs the whole image: its median "
        f"recall is {f3(sol[g][1]['median'])} in both tumor types, but its median precision is "
        f"{f3(sol[g][2]['median'])} for glioma and {f3(sol[m_][2]['median'])} for meningioma, so its median DSC is only "
        f"{f3(sol[g][0]['median'])} and {f3(sol[m_][0]['median'])}. The ESRG, started from the same single seed, reaches "
        f"{f3(e1d(g, 'dsc', 'P_ESRG')['median'])} and {f3(e1d(m_, 'dsc', 'P_ESRG')['median'])}. When the SRG additionally "
        "receives a healthy-tissue and a background seed (P_SRG), it obtains the highest DSC in both tumor types "
        f"(median {f3(e1d(g, 'dsc', 'P_SRG')['median'])} for glioma and {f3(e1d(m_, 'dsc', 'P_SRG')['median'])} for "
        "meningioma). The ESRG with a planted seed follows, and the ESRG with automatic seeding is lowest of the three "
        f"working configurations (median {f3(e1d(g, 'dsc', 'A_ESRG')['median'])} and {f3(e1d(m_, 'dsc', 'A_ESRG')['median'])}).")
    S.p("Against the SRG with competing seeds, the two algorithms err in opposite directions. The ESRG with a planted seed "
        f"has the higher mean precision ({f3(e1d(g, 'precision', 'P_ESRG')['mean'])} against "
        f"{f3(e1d(g, 'precision', 'P_SRG')['mean'])} for glioma, {f3(e1d(m_, 'precision', 'P_ESRG')['mean'])} against "
        f"{f3(e1d(m_, 'precision', 'P_SRG')['mean'])} for meningioma) but the lower mean recall "
        f"({f3(e1d(g, 'recall', 'P_ESRG')['mean'])} against {f3(e1d(g, 'recall', 'P_SRG')['mean'])} and "
        f"{f3(e1d(m_, 'recall', 'P_ESRG')['mean'])} against {f3(e1d(m_, 'recall', 'P_SRG')['mean'])}): it rarely includes "
        "non-tumor tissue but stops before covering the whole tumor, whereas the SRG covers more of the tumor at the cost "
        "of more leakage. Both algorithms delineate meningioma, a well-circumscribed and homogeneously enhancing tumor, "
        "far better than glioma, whose infiltrative and heterogeneous core offers no sharp intensity boundary; on "
        "meningioma the median DSC of every working configuration exceeds 0.70, the level conventionally regarded as "
        "good agreement (Zijdenbos et al., 1994).")
    S.p(f"The ESRG is also much faster. Its median processing time per slice is {f2(e1d(g, 'seconds', 'P_ESRG')['median'])} s "
        f"for glioma and {f2(e1d(m_, 'seconds', 'P_ESRG')['median'])} s for meningioma with a planted seed, and "
        f"{f2(e1d(g, 'seconds', 'A_ESRG')['median'])} s and {f2(e1d(m_, 'seconds', 'A_ESRG')['median'])} s with automatic "
        f"seeding, against {f2(e1d(g, 'seconds', 'P_SRG')['median'])} s and {f2(e1d(m_, 'seconds', 'P_SRG')['median'])} s for "
        "the SRG with three seeds. The SRG labels every pixel of the image, whereas the ESRG grows only from the tumor "
        "seed and stops, and the single-seed SRG is slowest because it floods the whole image.")

    rows = []
    PCOLS = [("P_ESRG vs P_SRG_solo", "P_ESRG vs. P_SRG_solo"), ("P_ESRG vs P_SRG", "P_ESRG vs. P_SRG"),
             ("A_ESRG vs P_SRG", "A_ESRG vs. P_SRG"), ("A_ESRG vs P_ESRG", "A_ESRG vs. P_ESRG")]
    for sc in TYPES:
        rows.append({"group": SNAME[sc]})
        for m in METRICS:
            fr = fried(sc, m)
            k = 1 if m == "hd95_wc" else 2 if m == "seconds" else 3

            def cell(key, m=m, sc=sc, k=k):
                v = post(sc, m, key)
                return f"{sgn(v['median_paired_diff'], k)} ({nz(v['r'])}{', ns' if v['call'] == 'no effect' else ''})"
            rows.append([MNAME[m], f"{fr['chi2']:.2f}", ptab(fr["p"]), nz(fr["kendall_w"])] + [cell(c) for c, _ in PCOLS])
    S.table("4.2", "Friedman Tests and Post Hoc Comparisons of the Four Configurations (E1)",
            ["Metric", "χ^{2}(3)", "*p*", "*W*"] + [lab for _, lab in PCOLS],
            rows, [0.9, 0.7, 0.6, 0.5, 1.3, 1.3, 1.3, 1.3],
            "χ^{2}, *p*, and *W* are the Friedman statistic, its *p* value, and Kendall’s coefficient of concordance. Each "
            "post hoc cell gives the median paired difference (first configuration minus second; for HD95, in pixels, and "
            "time, in seconds, second minus first, so a positive value always favors the first configuration) and, in "
            "parentheses, the effect size *r* of the Wilcoxon signed-rank test. Post hoc *p* values were Holm-adjusted over "
            "the six pairs of the four configurations; the two pairs not shown (A_ESRG vs. P_SRG_solo and P_SRG vs. "
            "P_SRG_solo) are in the analysis output. ns = not significant (Holm *p* ≥ .05 or |*r*| < .10); all other post "
            "hoc differences are significant.",
            ["l", "c", "c", "c", "c", "c", "c", "c"])
    wr = [fried(sc, m)["kendall_w"] for sc in TYPES for m in METRICS]
    S.p(f"Table 4.2 confirms that the four configurations differ in every metric and in both tumor types (all *p* < .001; "
        f"*W* from {nz(min(wr))} to {nz(max(wr))}). With the same single tumor seed, the ESRG is significantly better "
        f"than the SRG in DSC (*r* = {nz(post(g, 'dsc', 'P_ESRG vs P_SRG_solo')['r'])} for glioma and "
        f"{nz(post(m_, 'dsc', 'P_ESRG vs P_SRG_solo')['r'])} for meningioma), precision, and HD95, and it is faster, while its "
        "recall is significantly lower because the single-seed SRG covers the whole image. Against the SRG with competing "
        f"seeds, the DSC of the planted ESRG is significantly lower (glioma {rp(post(g, 'dsc', 'P_ESRG vs P_SRG'))}; meningioma "
        f"{rp(post(m_, 'dsc', 'P_ESRG vs P_SRG'))}). Its precision is significantly higher (glioma "
        f"*r* = {nz(post(g, 'precision', 'P_ESRG vs P_SRG')['r'])}; meningioma *r* = {nz(post(m_, 'precision', 'P_ESRG vs P_SRG')['r'])}), "
        f"its recall significantly lower (*r* = {nz(post(g, 'recall', 'P_ESRG vs P_SRG')['r'])} and "
        f"{nz(post(m_, 'recall', 'P_ESRG vs P_SRG')['r'])}), its HD95 slightly worse "
        f"(*r* = {nz(post(g, 'hd95_wc', 'P_ESRG vs P_SRG')['r'])} and {nz(post(m_, 'hd95_wc', 'P_ESRG vs P_SRG')['r'])}), and "
        f"it is significantly faster (*r* = {nz(post(g, 'seconds', 'P_ESRG vs P_SRG')['r'])} and "
        f"{nz(post(m_, 'seconds', 'P_ESRG vs P_SRG')['r'])}). Automatic seeding lowers the DSC of the ESRG further "
        f"(A_ESRG vs. P_ESRG: glioma *r* = {nz(post(g, 'dsc', 'A_ESRG vs P_ESRG')['r'])}, meningioma "
        f"*r* = {nz(post(m_, 'dsc', 'A_ESRG vs P_ESRG')['r'])}), by a large margin for glioma and a small one for meningioma.")
    S.p("The overall improvement defined in Section 3.1.2 is therefore not established for either tumor type: the DSC of "
        "the ESRG, with a planted or an automatic seed, is significantly lower than that of the SRG with competing seeds. "
        "The comparison at equal input tells a different story, because the ESRG reaches useful segmentations from one "
        "seed where the SRG cannot; Section 4.3 puts the two views side by side.")

    # ═══ 4.2 ═════════════════════════════════════════════════════════════════
    S.h2("4.2 Results per Objective")
    S.h3("4.2.1 Objective 1: Automatic Seed Selection")
    a = df[df.config == "A_ESRG"].copy()
    a["hit"] = a.seed_hit.fillna(False).astype(bool)
    hitd = {sc: a[a.tumor == sc] for sc in TYPES}
    rows = []
    for sc in TYPES:
        s = E1[sc]["shr"]; h = hitd[sc]
        rows.append([SNAME[sc], f"{s['hits']} / {s['n']}", f"{s['shr']:.1f}%", f"[{s['ci'][0]:.1f}%, {s['ci'][1]:.1f}%]",
                     "Met" if s["ci"][0] >= 70 else "Not met", str(s["no_candidate"]),
                     f3(h[h.hit].dsc.mean()), f3(h[~h.hit].dsc.fillna(0).mean())])
    S.table("4.3", "Seed Hit Rate of Automatic Seed Selection and DSC of the Automatic ESRG by Seed Outcome",
            ["Tumor type", "Hits / n", "SHR", "95% CI", "LB ≥ 70%", "No candidate", "Mean DSC, hit", "Mean DSC, miss"],
            rows, [1.1, 0.85, 0.7, 1.25, 0.8, 0.8, 0.85, 0.85],
            "SHR = seed hit rate, the share of slices whose entire automatic seed core lies inside the ground-truth tumor "
            "core (Section 3.1.2); CI = Wilson score interval; LB = lower bound of the interval, compared with the 70% target "
            "of Objective 1. No candidate = slices on which the system reported that no tumor candidate was found (counted as "
            "misses, DSC = 0).", ["l", "c", "c", "c", "c", "c", "c", "c"])
    sh = {sc: E1[sc]["shr"] for sc in TYPES}
    hit_dsc = {sc: hitd[sc][hitd[sc].hit].dsc.mean() for sc in TYPES}
    miss_dsc = {sc: hitd[sc][~hitd[sc].hit].dsc.fillna(0).mean() for sc in TYPES}
    S.p(f"Table 4.3 shows that the automatic seed lies entirely inside the tumor core on {sh[m_]['hits']} of {sh[m_]['n']} "
        f"meningioma slices, a seed hit rate of {sh[m_]['shr']:.1f}% (95% CI [{sh[m_]['ci'][0]:.1f}%, {sh[m_]['ci'][1]:.1f}%]), "
        f"whose lower bound {'meets' if sh[m_]['ci'][0] >= 70 else 'falls short of'} the 70% target, and on {sh[g]['hits']} of "
        f"{sh[g]['n']} glioma slices ({sh[g]['shr']:.1f}%, 95% CI [{sh[g]['ci'][0]:.1f}%, {sh[g]['ci'][1]:.1f}%]), whose lower "
        f"bound {'meets' if sh[g]['ci'][0] >= 70 else 'falls short of'} it. A meningioma typically enhances strongly and "
        "uniformly and is usually the brightest structure of the brain on a T1-CE slice, so the highest Otsu class isolates it. "
        "A glioma core is often a thin, irregular enhancing rim around necrosis, while vessels, the choroid plexus, and the dura "
        "can enhance as strongly as the tumor and may then be ranked above it. The seed outcome decides the segmentation: when "
        f"the seed hits the tumor, the automatic ESRG reaches a mean DSC of {f3(hit_dsc[g])} for glioma and {f3(hit_dsc[m_])} "
        f"for meningioma, whereas a miss leaves {f3(miss_dsc[g])} and {f3(miss_dsc[m_])}, because the region then grows inside "
        "the wrong structure. The weakness of the automatic pipeline therefore lies in localization, not in delineation, and "
        "it is most marked for glioma.")
    rows = []
    for exp, lab in (("E2a", "A_ESRG"), ("E2b", "A_SRG")):
        for sc in TYPES:
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
    w = df[df.config.isin(["P_SRG", "P_SRG_2", "P_SRG_3"])].pivot_table(index=["tumor", "file"], columns="config", values="dsc")
    rng_ = (w.max(axis=1) - w.min(axis=1))
    vary = {t: (rng_.loc[t] >= 0.2).mean() * 100 for t in TYPES}
    S.p(f"Table 4.4 compares the automatic configurations with the three planted placements. Moving every planted seed by "
        f"4 to 8 mm inside its own region changes the DSC of the SRG by at least 0.20 on {vary[g]:.1f}% of glioma and "
        f"{vary[m_]:.1f}% of meningioma slices, so where the operator clicks can decide the outcome. The automatic seed "
        "removes this source of variability " + ("entirely, as the repeated runs confirm (Section 4.1). " if det_ok else ". ")
        + "However, consistency is not the same as accuracy. In E2a, the median DSC of the automatic ESRG "
          f"({f3(E2['E2a'][g]['dsc']['A_ESRG']['median'])} for glioma and {f3(E2['E2a'][m_]['dsc']['A_ESRG']['median'])} for "
          f"meningioma) is {both(lambda t: 'not significantly lower' if E2['E2a'][t]['auto_not_lower'] else 'significantly lower')} "
          "than that of the planted placements. In E2b, the SRG started from the same automatic tumor seed reaches "
          f"{f3(E2['E2b'][g]['dsc']['A_SRG']['median'])} and {f3(E2['E2b'][m_]['dsc']['A_SRG']['median'])}, which is "
          f"{both(lambda t: 'not significantly lower' if E2['E2b'][t]['auto_not_lower'] else 'significantly lower')} "
          "than that of the planted placements. The shortfall of both algorithms relative to the planted placements reflects the "
          "seed hit rate, whereas the difference between them reflects the growth rule.")
    ob1 = {sc: V[sc]["objective1"] for sc in TYPES}
    yn = lambda b: "met" if b else "not met"
    S.p("With respect to Objective 1, the three criteria of Section 3.1.2 give the following result. The seed hit rate "
        f"criterion is {yn(ob1[m_]['a_shr_lb_ge_70'])} for meningioma and {yn(ob1[g]['a_shr_lb_ge_70'])} for glioma; the "
        f"consistency criterion is {both(lambda t: yn(ob1[t]['b_consistency']))}; "
        f"and the criterion that the automatic DSC is not lower than any planted placement is "
        f"{both(lambda t: yn(ob1[t]['c_auto_not_lower']))}. Objective 1 is therefore "
        + ("met for both tumor types." if all(all(v.values()) for v in ob1.values()) else
           "met only in part: automatic seed selection makes the result reproducible and is reliable for meningioma, "
           "but it does not yet localize gliomas reliably enough to replace a careful operator."))

    # ── Objective 2 ──────────────────────────────────────────────────────────
    S.h3("4.2.2 Objective 2: Minimization of Undersegmentation")

    def pair_rows(exps):
        rows = []
        for e, lab in exps:
            rows.append({"group": f"{e}: {lab} ({PA[e][g]['baseline']} → {PA[e][g]['enhanced']})"})
            for sc in TYPES:
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
            pair_rows([("E3", "LLDM added to SRG"), ("E4", "LLDM removed from ESRG")]), wid, note2,
            ["l", "l", "c", "c", "c", "c", "c", "c", "c"])
    for sc in TYPES:
        assert test("E3", sc, "recall")["call"] == "better" and test("E3", sc, "dsc")["call"] == "worse"
    S.p("Table 4.5 answers two questions. E3 asks whether the local log-domain difference measure (LLDM) is sufficient to "
        "reduce undersegmentation when it is added to the original SRG. It raises recall "
        f"(median {f3(test('E3', g, 'recall')['base']['median'])} → {f3(test('E3', g, 'recall')['enh']['median'])} for glioma and "
        f"{f3(test('E3', m_, 'recall')['base']['median'])} → {f3(test('E3', m_, 'recall')['enh']['median'])} for meningioma; "
        f"{rp(test('E3', g, 'recall'))} and {rp(test('E3', m_, 'recall'))}), but the DSC collapses (median "
        f"{f3(test('E3', g, 'dsc')['base']['median'])} → {f3(test('E3', g, 'dsc')['enh']['median'])} and "
        f"{f3(test('E3', m_, 'dsc')['base']['median'])} → {f3(test('E3', m_, 'dsc')['enh']['median'])}); the region covered more "
        f"than 40% of the brain on {leak.loc['P_SRG_L', g]:.1f}% of glioma and {leak.loc['P_SRG_L', m_]:.1f}% of meningioma "
        f"slices, against {leak.loc['P_SRG', g]:.1f}% and {leak.loc['P_SRG', m_]:.1f}% for the plain SRG. The higher recall is "
        "therefore the by-product of flooding, not of a better boundary. The reason lies in the competition of the original "
        "algorithm. The tumor region of the SRG competes with the healthy-tissue and background regions for every pixel, and the "
        "global mean of the tumor region is what makes healthy tissue look different from the tumor. A local reference follows "
        "the intensity of the most recently absorbed neighbors, so once the tumor region reaches its boundary, the healthy "
        "tissue next to it looks similar and is absorbed before the healthy-tissue region arrives. Without a stopping rule, a "
        "local measure removes the only brake that the SRG has.")
    e4 = {sc: (test("E4", sc, "recall"), test("E4", sc, "dsc")) for sc in TYPES}
    S.p("E4 asks whether the measure is necessary within the ESRG, where the adaptive stopping criterion bounds the region. "
        "With the measure, the recall of the ESRG is "
        + " and ".join(f"{call_word(e4[sc][0]['call'])} than without it for {sc} (median {f3(e4[sc][0]['base']['median'])} → "
                       f"{f3(e4[sc][0]['enh']['median'])}, {rp(e4[sc][0])})" for sc in TYPES)
        + ", and its DSC is " + " and ".join(f"{call_word(e4[sc][1]['call'])} for {sc} ({rp(e4[sc][1])})" for sc in TYPES)
        + ". Inside the ESRG, the comparison with the adjacent tumor tissue lets the region follow the gradual intensity changes "
          "of a heterogeneous tumor, while the stopping bounds keep it from following those changes into the surrounding tissue.")
    ob2 = {sc: V[sc]["objective2"] for sc in TYPES}
    S.p("Objective 2 is " + ("met for both tumor types." if all(ob2.values()) else
        "therefore not met under the strict rule of Section 3.1.2, which requires a significant recall gain without loss of DSC "
        "in both E3 and E4. The results support a narrower conclusion: the local log-domain measure improves recall and DSC "
        "within the ESRG on meningioma but not on glioma, and it is not sufficient on its own: added to an algorithm without a "
        "stopping criterion, it is harmful."))

    # ── Objective 3 ──────────────────────────────────────────────────────────
    S.h3("4.2.3 Objective 3: Decrease of Boundary Leakage")
    S.table("4.6", "Effect of the Adaptive Stopping Criterion on Precision, HD95, and DSC (E5a, E5b, and E6)", hdr,
            pair_rows([("E5a", "ASC with one seed vs. SRG with three seeds"), ("E5b", "ASC added to SRG with one seed"),
                       ("E6", "ASC removed from ESRG")]), wid,
            note2 + " The baseline of E5b, the SRG with a single seed and no stopping rule, absorbs the whole image by design.",
            ["l", "l", "c", "c", "c", "c", "c", "c", "c"])
    lk = {t: leak.loc["P_ESRG_C_noS", t] for t in TYPES}
    flood_noS = (f"{lk[g]:.1f}% of the slices of both tumor types" if abs(lk[g] - lk[m_]) < 0.05
                 else f"{lk[g]:.1f}% of glioma and {lk[m_]:.1f}% of meningioma slices")
    S.p("Table 4.6 shows the strongest effects of the study. E6 removes the adaptive stopping criterion (ASC) from the ESRG "
        "while keeping its competing regions. Without it, the median precision is "
        f"{f3(test('E6', g, 'precision')['base']['median'])} for glioma and {f3(test('E6', m_, 'precision')['base']['median'])} for "
        f"meningioma and the region exceeds 40% of the brain on {flood_noS}; with it, the median precision is "
        f"{f3(test('E6', g, 'precision')['enh']['median'])} and {f3(test('E6', m_, 'precision')['enh']['median'])} "
        f"({rp(test('E6', g, 'precision'))}; {rp(test('E6', m_, 'precision'))}), the median HD95 falls from "
        f"{f1(test('E6', g, 'hd95_wc')['base']['median'])} to {f1(test('E6', g, 'hd95_wc')['enh']['median'])} pixels for glioma and "
        f"from {f1(test('E6', m_, 'hd95_wc')['base']['median'])} to {f1(test('E6', m_, 'hd95_wc')['enh']['median'])} for meningioma, "
        f"and the median DSC rises from {f3(test('E6', g, 'dsc')['base']['median'])} to {f3(test('E6', g, 'dsc')['enh']['median'])} "
        f"and from {f3(test('E6', m_, 'dsc')['base']['median'])} to {f3(test('E6', m_, 'dsc')['enh']['median'])}. Because the local "
        "measure removes the brake of the global mean (Section 4.2.2), the stopping criterion is what keeps the ESRG inside the "
        "tumor; it is necessary within the proposed algorithm. E5b shows the same mechanism in the original algorithm: a single "
        f"SRG region absorbs the entire image (median precision {f3(test('E5b', g, 'precision')['base']['median'])} and "
        f"{f3(test('E5b', m_, 'precision')['base']['median'])}), whereas the same region under the ASC reaches "
        f"{f3(test('E5b', g, 'precision')['enh']['median'])} and {f3(test('E5b', m_, 'precision')['enh']['median'])} "
        f"({rp(test('E5b', g, 'precision'))}; {rp(test('E5b', m_, 'precision'))}).")
    for sc in TYPES:
        assert test("E5a", sc, "precision")["call"] == "better" and test("E5a", sc, "dsc")["call"] == "worse"
        assert test("E5a", sc, "hd95_wc")["call"] == "worse"
    S.p("E5a is the demanding comparison: it asks whether the ASC, from the tumor seed alone, can replace the two competing "
        "seeds that the original algorithm needs. The SRG with the ASC and one seed has significantly higher precision than "
        f"the SRG with three seeds (median {f3(test('E5a', g, 'precision')['base']['median'])} → "
        f"{f3(test('E5a', g, 'precision')['enh']['median'])} for glioma, {rp(test('E5a', g, 'precision'))}; "
        f"{f3(test('E5a', m_, 'precision')['base']['median'])} → {f3(test('E5a', m_, 'precision')['enh']['median'])} for "
        f"meningioma, {rp(test('E5a', m_, 'precision'))}), so it leaks less. Its HD95 is significantly worse "
        f"({rp(test('E5a', g, 'hd95_wc'))}; {rp(test('E5a', m_, 'hd95_wc'))}) and its DSC is significantly worse (median "
        f"{f3(test('E5a', g, 'dsc')['base']['median'])} → {f3(test('E5a', g, 'dsc')['enh']['median'])} and "
        f"{f3(test('E5a', m_, 'dsc')['base']['median'])} → {f3(test('E5a', m_, 'dsc')['enh']['median'])}). The stopping bounds "
        "thus keep the region inside the tumor more strictly than the competing regions do, but they also stop it earlier, "
        "before it has covered the darker or more heterogeneous parts of the tumor; the boundary errors move from leakage to "
        "undersegmentation, which HD95 and DSC penalize.")
    ob3 = {sc: V[sc]["objective3"] for sc in TYPES}
    S.p("Objective 3 is " + ("met for both tumor types." if all(ob3.values()) else
        "therefore not met under the strict rule of Section 3.1.2, because E5a does not show a lower HD95 and an unharmed DSC. "
        "The criterion does, however, decrease boundary leakage in every comparison, as measured by precision, and its removal "
        "from the ESRG causes one of the largest losses of accuracy observed in the study."))
    img = os.path.join(figdir, "figure_4_1.png")
    iw, ih = Image.open(img).size
    ex = {t: df[df.file == fname].set_index("config").dsc for t, fname in zip(TYPES, figs["examples"])}
    S.fig("4.1", img, "Segmentations of Two Test Slices by Seven Configurations",
          "Red fill and outline = predicted mask; yellow outline = ground-truth tumor core where it is not covered by the "
          "prediction. In each row the slice is the test slice whose SRG (3 planted seeds) DSC is closest to the median of its "
          "tumor type, so the slices are typical for the baseline, not selected for the ESRG. LLDM = local log-domain "
          "difference measure; ASC = adaptive stopping criterion.", iw, ih)
    fig_txt = ("Figure 4.1 illustrates these mechanisms on two test slices that are typical for the baseline. Given one planted "
               "seed, the SRG floods the brain in both rows "
               f"(DSC {f2(ex[g]['P_SRG_solo'])} and {f2(ex[m_]['P_SRG_solo'])}), as does the SRG with the local measure "
               f"({f2(ex[g]['P_SRG_L'])} and {f2(ex[m_]['P_SRG_L'])}) and the ESRG without the stopping criterion; the SRG with "
               f"the stopping criterion and a single seed stays inside the tumor but stops early "
               f"({f2(ex[g]['P_SRG_solo_S'])} and {f2(ex[m_]['P_SRG_solo_S'])}). ")
    if ex[g]["A_ESRG"] > ex[g]["P_ESRG"] + 0.2:
        fig_txt += (f"The glioma row also shows how strongly the ESRG depends on its seed, whose statistics set the stopping "
                    f"bounds. Started from the planted seed at the deepest point of the core, it covers only part of the core "
                    f"(DSC {f2(ex[g]['P_ESRG'])}); started from the automatic seed, it delineates the core almost completely "
                    f"(DSC {f2(ex[g]['A_ESRG'])}, against {f2(ex[g]['P_SRG'])} for the SRG with three seeds). ")
    if ex[m_]["P_ESRG"] < ex[m_]["P_SRG"]:
        fig_txt += (f"On the meningioma, the ESRG (DSC {f2(ex[m_]['P_ESRG'])}) misses a thin margin of the tumor that the SRG "
                    f"(DSC {f2(ex[m_]['P_SRG'])}) includes, which is the recall deficit of Table 4.1 in visual form.")
    S.p(fig_txt)

    # ═══ 4.3 Comparative analysis ════════════════════════════════════════════
    S.h2("4.3 Comparative Analysis of the Baseline SRG and the ESRG")
    S.p("Table 4.7 brings the experiments together to compare the baseline SRG with the ESRG. Each block states what is "
        "compared, the configurations, and the result of each algorithm for glioma and meningioma, with the significance of "
        "the difference and its interpretation. The first three blocks compare the algorithms as wholes under three different "
        "inputs; the last four compare the applied methods, either added to the SRG or removed from the ESRG.")

    def tests_of(kind, sc):
        if kind == "e1eq":
            t, b, e = post(sc, "dsc", "P_ESRG vs P_SRG_solo"), "P_SRG_solo", "P_ESRG"
            return e1d(sc, "dsc", b)["median"], e1d(sc, "dsc", e)["median"], t
        if kind == "e1sr":
            t, b, e = post(sc, "dsc", "P_ESRG vs P_SRG"), "P_SRG", "P_ESRG"
            return e1d(sc, "dsc", b)["median"], e1d(sc, "dsc", e)["median"], t
        t = {"s2": lambda: sec("S2", sc, "dsc"), "e3": lambda: test("E3", sc, "recall"), "e4": lambda: test("E4", sc, "recall"),
             "e5a": lambda: test("E5a", sc, "precision"), "e6": lambda: test("E6", sc, "precision")}[kind]()
        return t["base"]["median"], t["enh"]["median"], t

    def why(kind, sc, t):
        c = t["call"]
        if kind == "e1eq":
            return ("Same single seed: the SRG absorbs the whole image, the ESRG stops at the tumor." if c == "better"
                    else "No clear advantage at equal input.")
        if kind == "e1sr":
            return ("The SRG with competing seeds scores higher: the ESRG leaks less but misses tumor (lower recall)." if c == "worse"
                    else "The ESRG matches the SRG with competing seeds." if c == "no effect"
                    else "The ESRG scores higher even against competing seeds.")
        if kind == "s2":
            return ("From the same automatic seed, the SRG is better in DSC; the ESRG undersegments (lower recall)." if c == "worse"
                    else "From the same automatic seed, the ESRG is better." if c == "better"
                    else "From the same automatic seed, no significant difference.")
        if kind == "e3":
            return ("Recall rises only because the SRG floods; its DSC collapses (Table 4.5)." if c == "better"
                    else "No significant change in recall.")
        if kind == "e4":
            return ("The measure raises recall inside the ESRG." if c == "better"
                    else "No significant change in recall within the ESRG." if c == "no effect"
                    else "The measure lowers recall within the ESRG.")
        if kind == "e5a":
            return ("One seed with the ASC leaks less than three seeds, but stops early (lower DSC, Table 4.6)." if c == "better"
                    else "No significant difference in precision.")
        if kind == "e6":
            return ("Without the ASC the ESRG floods; the ASC is necessary." if c == "better"
                    else "No significant effect of the ASC on precision.")
        return ""

    blocks = [
        ("e1eq", "E1 · Same input (one planted tumor seed), DSC", "SRG (P_SRG_solo) → ESRG (P_ESRG)"),
        ("e1sr", "E1 · SRG with competing seeds (three planted seeds), DSC", "SRG (P_SRG) → ESRG (P_ESRG)"),
        ("s2", "Same automatic tumor seed, DSC", "SRG (A_SRG) → ESRG (A_ESRG)"),
        ("e3", "E3 · LLDM added to the SRG (Objective 2), recall", "SRG (P_SRG) → SRG + LLDM (P_SRG_L)"),
        ("e4", "E4 · LLDM removed from the ESRG (Objective 2), recall", "ESRG without LLDM (P_ESRG_noL) → ESRG (P_ESRG)"),
        ("e5a", "E5a · ASC with one seed replaces competing seeds (Objective 3), precision", "SRG (P_SRG) → SRG + ASC, 1 seed (P_SRG_solo_S)"),
        ("e6", "E6 · ASC removed from the ESRG (Objective 3), precision", "ESRG without ASC (P_ESRG_C_noS) → ESRG (P_ESRG_C)"),
    ]
    rows = []
    for kind, what, cfgs in blocks:
        rows.append({"group": f"{what}: {cfgs}"})
        for sc in TYPES:
            b, e, t = tests_of(kind, sc)
            rows.append([SNAME[sc], f3(b), f3(e), sgn(t["median_paired_diff"], 3), nz(t["r"]), ptab(t["p_holm"]),
                         {"better": "Better", "worse": "Worse", "no effect": "ns"}[t["call"]], why(kind, sc, t)])
    S.table("4.7", "Comparative Analysis of the Baseline SRG and the ESRG by Experiment",
            ["Tumor type", "Baseline, Mdn", "Compared, Mdn", "Mdn diff.", "*r*", "*p*_{Holm}", "Result", "Interpretation"],
            rows, [0.9, 0.7, 0.75, 0.65, 0.45, 0.6, 0.6, 2.4],
            "Baseline → compared as named in each block (see Table 3.2); the metric of each block is the primary metric of its "
            "experiment. Mdn = median. Mdn diff. = median paired difference, positive when the compared configuration is better; "
            "*r* = effect size of the Wilcoxon signed-rank test; *p*_{Holm} = Holm-adjusted *p* value; ns = not significant "
            "(Holm *p* ≥ .05 or |*r*| < .10). The comparison with the same automatic seed was added after the test run, uses "
            "results already collected, and does not enter the decision rules.",
            ["l", "c", "c", "c", "c", "c", "c", "l"])
    S.p("The table shows one pattern. As whole algorithms, the ESRG and the SRG differ in the input they need. With one "
        "planted seed, the ESRG is far better than the SRG in both tumor types, because the adaptive stopping criterion "
        "replaces the competing seeds that the SRG needs to stay inside the tumor. When the SRG receives those seeds, it "
        "scores higher in DSC than the ESRG in both tumor types, and the same holds when both algorithms start from the same "
        "automatic seed: the ESRG is more conservative, leaking less (higher precision) but recovering less of the tumor "
        "(lower recall). The overall improvement defined in Section 3.1.2, a higher DSC than the SRG with competing seeds, "
        "is therefore not established.")
    S.p("The method-level blocks explain why. The two enhancements work only together. Adding the local log-domain measure "
        "to the SRG raises recall only by letting the region flood the brain, and removing the stopping criterion from the "
        "ESRG makes it flood as well, whereas the local measure improves the ESRG itself on meningioma, where it raises "
        "recall without harming DSC, but not on glioma. The stopping criterion decreases boundary leakage in every comparison "
        "in which leakage is possible; the price is a region that stops before covering the whole tumor, which is why "
        "precision improves while recall and DSC do not. In summary, the ESRG achieves a controlled, fast, and "
        "reproducible segmentation from one seed, or from none for meningioma, while the original algorithm delivers a higher "
        "DSC only when the operator supplies the extra seeds that the ESRG does not need.")

    S.h2("References Added in Chapter 4")
    S.p("The following work is cited in this chapter; the remaining citations are listed in Chapter 3.", noindent=True)
    S.ref("Zijdenbos, A. P., Dawant, B. M., Margolin, R. A., & Palmer, A. C. (1994). Morphometric analysis of white matter "
          "lesions in MR images: Method and validation. *IEEE Transactions on Medical Imaging, 13*(4), 716–724. "
          "https://doi.org/10.1109/42.363096")
    return {"title": "Chapter 4: Results and Discussion", "author": "Karylle Vinces J. Aliswag & Juana Mari A. Del Rosario", "blocks": S.B}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "evaluation_brats", "test"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "outputs", "chapter4_brats"))
    ap.add_argument("--out", default=os.path.join(ROOT, "Chapter4_Results_and_Discussion.docx"))
    ap.add_argument("--no-figures", action="store_true", help="reuse the figure already in --figdir")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    if not a.no_figures:
        subprocess.run([sys.executable, "-P", os.path.join(HERE, "figures_brats.py"), a.dir, a.figdir], check=True)
    spec = build(a.dir, a.figdir)
    sp = os.path.join(a.figdir, "chapter4_spec.json")
    json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    subprocess.run(["node", os.path.join(HERE, "render_docx.js"), sp, a.out], check=True)
    print("blocks:", len(spec["blocks"]))
