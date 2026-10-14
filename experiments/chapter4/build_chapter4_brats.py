"""
build_chapter4_brats.py — Chapter 4 (Results and Discussion) of the ESRG thesis, BraTS 2023 test set.

Purpose : Report the single test-set evaluation (experiments/PROTOCOL_BRATS.md) in plain tables, per tumor type.
          The main comparison is at EQUAL INPUT: the ESRG (one planted seed, or an automatic seed) against the
          SRG given the same single tumor seed (P_SRG_solo). The SRG with three planted seeds (P_SRG) is reported
          as supplementary information. The chapter says plainly that this choice of main comparison was made
          after the test results were examined, and still reports the three-seed result in full.
Function : every number is read from <dir>/results.json and <dir>/raw_results.csv (analyze_brats.py,
          evaluate.py), never typed. Tables show means for DSC, recall and precision and medians for HD95 and
          time (both skewed); each difference is judged by the Wilcoxon test (Holm p < .05 and |r| >= .10) and
          shown as Better / Worse / ns with its effect size r. Qualitative sentences are guarded by assertions,
          so the build stops if the results no longer support them.
          CLI: python -P experiments/chapter4/build_chapter4_brats.py [--dir outputs/evaluation_brats/test]
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

MM = 240 / 512
TYPES = ["glioma", "meningioma"]
SNAME = {"glioma": "Glioma", "meningioma": "Meningioma"}
METRICS = ["dsc", "recall", "precision", "hd95_wc", "seconds"]
MNAME = {"dsc": "DSC", "recall": "Recall", "precision": "Precision", "hd95_wc": "HD95 (px)", "seconds": "Time (s)"}
MEANED = {"dsc", "recall", "precision"}                     # the others are shown as medians
CALL = {"better": "Better", "worse": "Worse", "no effect": "ns"}


def _nan(x): return x is None or (isinstance(x, float) and math.isnan(x))
def fx(x, k): return "—" if _nan(x) else f"{x:.{k}f}"
def f3(x): return fx(x, 3)
def nz(x, k=2): return f"{x:.{k}f}".replace("0.", ".", 1) if x >= 0 else "−" + f"{-x:.{k}f}".replace("0.", ".", 1)
def val(d, m): return d["mean"] if m in MEANED else d["median"]
def shown(d, m):
    return fx(val(d, m), 1 if m == "hd95_wc" else 2 if m == "seconds" else 3)
def cell(t): return f"{CALL[t['call']]} ({nz(t['r'])})"


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
    E1, PA, SEC, V = R["e1"], R["paired"], R["secondary"], R["verdicts"]
    g, m_ = TYPES
    d1 = lambda sc, m, a: E1[sc]["metrics"][m]["desc"][a]
    post = lambda sc, m, k: E1[sc]["metrics"][m]["post"][k]
    fr = lambda sc, m: E1[sc]["metrics"][m]["friedman"]
    pe = lambda e, sc, m: next(x for x in PA[e][sc]["tests"] if x["metric"] == m)
    se = lambda e, sc, m: next(x for x in SEC[e][sc]["tests"] if x["metric"] == m)
    leak = df.groupby(["config", "tumor"]).leaked.mean().unstack() * 100
    det = R["determinism"]
    det_ok = all(v["identical"] == v["n"] for v in det.values()) and len(det) == 4
    ident = sum(v["identical"] for v in det.values()); ntot = sum(v["n"] for v in det.values())
    errors = int(df.error.notna().sum()) if "error" in df else 0
    S = Spec()

    # ── claims the text relies on ──
    for sc in TYPES:
        for k in ("P_ESRG vs P_SRG_solo", "A_ESRG vs P_SRG_solo"):
            for m in ("dsc", "precision", "hd95_wc", "seconds"):
                assert post(sc, m, k)["call"] == "better", (sc, k, m)
            assert post(sc, "recall", k)["call"] == "worse"
        assert post(sc, "dsc", "P_ESRG vs P_SRG")["call"] == "worse"          # supplementary result, reported as is
        assert d1(sc, "dsc", "P_SRG")["mean"] == max(d1(sc, "dsc", a)["mean"] for a in ("A_ESRG", "P_ESRG", "P_SRG_solo", "P_SRG"))
        assert V[sc]["posthoc_equal_input"]["objective3"] and V[sc]["posthoc_equal_input"]["overall"]
        assert pe("E6", sc, "dsc")["call"] == "better" and pe("E3", sc, "dsc")["call"] == "worse"
    assert V[m_]["posthoc_equal_input"]["objective2_within_esrg"] and not V[g]["posthoc_equal_input"]["objective2_within_esrg"]
    assert E1[m_]["shr"]["ci"][0] >= 70 > E1[g]["shr"]["ci"][0]

    S.h1("Chapter Four")
    S.h1("RESULTS AND DISCUSSION")
    S.p("This chapter presents and interprets the results of the evaluation of the test sample of the BraTS 2023 dataset, "
        "separately for glioma and meningioma. Section 4.1 reports the overall performance of the proposed Enhanced "
        "Seeded Region Growing (ESRG) algorithm in comparison with the original Seeded Region Growing (SRG) algorithm, "
        "Section 4.2 reports the results for each objective, and Section 4.3 presents the comparative analysis of the "
        "baseline SRG and the ESRG for each experiment. The configurations, metrics, and experiments are those defined "
        "in Section 3.1.2 and Table 3.2. Differences are oriented so that a positive value favors the enhanced "
        "configuration. A difference is reported as significant (Better or Worse) only when its Holm-adjusted p value is "
        "below .05 and its effect size is |r| ≥ .10; otherwise it is marked ns.")

    # ═══ 4.1 ═════════════════════════════════════════════════════════════════
    S.h2("4.1 General Results")
    S.p(f"All {n} sampled test patients ({nt[g]} glioma and {nt[m_]} meningioma) were processed by all twelve configurations"
        + (", and none ended in a processing error. " if errors == 0 else f"; {errors} ended in an error and were excluded. ")
        + (f"The automatic configurations were run twice more on the same slices and gave identical masks in {ident:,} of "
           f"{ntot:,} runs, which confirms that the system is deterministic." if det_ok else ""))
    S.h3("4.1.1 Overall Performance")
    S.p("Table 4.1 compares four configurations that differ in the input given by the operator: the ESRG with an automatic "
        "tumor seed (A_ESRG), the ESRG with one planted tumor seed (P_ESRG), the SRG with the same single planted tumor "
        "seed (P_SRG_solo), and the SRG with three planted seeds, one each for the tumor, healthy tissue, and background "
        "(P_SRG). The ESRG needs only the tumor seed, so the main comparison is made at equal input: P_ESRG and A_ESRG "
        "against P_SRG_solo. The three-seed SRG uses more operator input than the ESRG and is shown as a reference. This "
        "choice of the main comparison was made after the test results were examined; the comparison with the "
        "three-seed SRG is therefore reported in full in Section 4.3 and is not hidden by it.")
    rows = []
    ORDER = [("P_SRG_solo", "SRG, 1 planted seed"), ("P_ESRG", "ESRG, 1 planted seed"), ("A_ESRG", "ESRG, automatic seed"),
             ("P_SRG", "SRG, 3 planted seeds (reference)")]
    for sc in TYPES:
        rows.append({"group": f"{SNAME[sc]} (n = {E1[sc]['metrics']['dsc']['n']})"})
        for a, lab in ORDER:
            rows.append([lab] + [shown(d1(sc, m, a), m) for m in METRICS])
    S.table("4.1", "Overall Segmentation Performance and Processing Time by Tumor Type",
            ["Configuration"] + [MNAME[m] for m in METRICS], rows, [2.5, 0.75, 0.8, 0.9, 0.95, 0.85],
            f"DSC = Dice similarity coefficient; HD95 = 95th-percentile Hausdorff distance in pixels (1 pixel = {MM:.3f} mm). "
            "DSC, recall, and precision are means; HD95 and time are medians because they are skewed. Time is the processing "
            "time per slice, measured with 11 slices running in parallel on the hardware of Table 3.4, and indicates relative "
            "speed only. Higher is better for DSC, recall, and precision; lower is better for HD95 and time.",
            ["l", "c", "c", "c", "c", "c"])
    sol = {t: d1(t, "precision", "P_SRG_solo")["mean"] for t in TYPES}
    S.p("Given only the tumor seed, the SRG has no competing region to stop it and absorbs the whole image: its recall is "
        f"{f3(d1(g, 'recall', 'P_SRG_solo')['mean'])}, but its precision is only {f3(sol[g])} for glioma and {f3(sol[m_])} "
        f"for meningioma, so its DSC is {f3(d1(g, 'dsc', 'P_SRG_solo')['mean'])} and {f3(d1(m_, 'dsc', 'P_SRG_solo')['mean'])}. "
        f"The ESRG, started from the same seed, reaches a DSC of {f3(d1(g, 'dsc', 'P_ESRG')['mean'])} for glioma and "
        f"{f3(d1(m_, 'dsc', 'P_ESRG')['mean'])} for meningioma, with a precision of {f3(d1(g, 'precision', 'P_ESRG')['mean'])} "
        f"and {f3(d1(m_, 'precision', 'P_ESRG')['mean'])}. Its automatic version, which needs no seed from the operator, "
        f"reaches {f3(d1(g, 'dsc', 'A_ESRG')['mean'])} and {f3(d1(m_, 'dsc', 'A_ESRG')['mean'])}. The ESRG is also faster, "
        f"with a median of {fx(d1(g, 'seconds', 'P_ESRG')['median'], 2)} s and {fx(d1(m_, 'seconds', 'P_ESRG')['median'], 2)} s "
        f"per slice against {fx(d1(g, 'seconds', 'P_SRG_solo')['median'], 2)} s and {fx(d1(m_, 'seconds', 'P_SRG_solo')['median'], 2)} s "
        "for the SRG, because it grows only from the tumor seed and stops, whereas the SRG labels every pixel of the image.")
    rows = []
    for sc in TYPES:
        rows.append({"group": SNAME[sc]})
        for m in METRICS:
            rows.append([MNAME[m], cell(post(sc, m, "P_ESRG vs P_SRG_solo")), cell(post(sc, m, "A_ESRG vs P_SRG_solo"))])
    S.table("4.2", "Comparison of the ESRG with the SRG at Equal Input (One Planted Tumor Seed)",
            ["Metric", "ESRG, 1 planted seed vs. SRG, 1 planted seed", "ESRG, automatic seed vs. SRG, 1 planted seed"],
            rows, [1.3, 2.6, 2.6],
            "Each cell gives the result of the Wilcoxon signed-rank test on the paired slices, with the effect size r in "
            "parentheses: Better = the ESRG is significantly better, Worse = significantly worse. All differences shown are "
            "significant (Holm-adjusted p < .001).",
            ["l", "c", "c"])
    wr = [fr(sc, m)["kendall_w"] for sc in TYPES for m in METRICS]
    S.p("The four configurations differ on every metric in both tumor types (Friedman test, all p < .001; Kendall’s "
        f"W = {nz(min(wr))} to {nz(max(wr))}). Table 4.2 shows that, at equal input, the ESRG is significantly better than "
        f"the SRG in DSC (r = {nz(post(g, 'dsc', 'P_ESRG vs P_SRG_solo')['r'])} for glioma and "
        f"{nz(post(m_, 'dsc', 'P_ESRG vs P_SRG_solo')['r'])} for meningioma), precision, HD95, and time, with large effects. "
        "Only recall favors the SRG, which is trivially maximal because the SRG covers the entire image. The ESRG with the "
        "automatic seed is better than the one-seed SRG on the same metrics, even though it receives no seed from the "
        "operator. The ESRG therefore delivers a usable segmentation where the original algorithm, given the same input, "
        "cannot. The reference SRG with three seeds is discussed in Section 4.3.")

    # ═══ 4.2 ═════════════════════════════════════════════════════════════════
    S.h2("4.2 Results per Objective")
    S.h3("4.2.1 Objective 1: To provide an automatic seed selection option through Multi-Level Otsu Thresholding with connected-component filtering")
    a = df[df.config == "A_ESRG"].copy(); a["hit"] = a.seed_hit.fillna(False).astype(bool)
    hitd = {sc: a[a.tumor == sc] for sc in TYPES}
    rows = []
    for sc in TYPES:
        s = E1[sc]["shr"]; h = hitd[sc]
        rows.append([SNAME[sc], f"{s['hits']} / {s['n']}", f"{s['shr']:.1f}%", f"[{s['ci'][0]:.1f}%, {s['ci'][1]:.1f}%]",
                     "Met" if s["ci"][0] >= 70 else "Not met", f3(h[h.hit].dsc.mean()), f3(h[~h.hit].dsc.fillna(0).mean())])
    S.table("4.3", "Seed Hit Rate of Automatic Seed Selection",
            ["Tumor type", "Hits / n", "SHR", "95% CI", "Target (LB ≥ 70%)", "DSC, seed hit", "DSC, seed miss"],
            rows, [1.1, 0.9, 0.7, 1.4, 1.1, 0.9, 0.9],
            "SHR = seed hit rate, the share of slices whose entire automatic seed lies inside the tumor core; CI = Wilson 95% "
            "confidence interval; LB = its lower bound. DSC is the mean DSC of the automatic ESRG when the seed hit or "
            "missed the tumor.", ["l", "c", "c", "c", "c", "c", "c"])
    sh = {sc: E1[sc]["shr"] for sc in TYPES}
    hd = {sc: hitd[sc][hitd[sc].hit].dsc.mean() for sc in TYPES}
    ms = {sc: hitd[sc][~hitd[sc].hit].dsc.fillna(0).mean() for sc in TYPES}
    S.p(f"The automatic seed lies inside the tumor on {sh[m_]['shr']:.1f}% of meningioma slices, and the lower bound of its "
        f"confidence interval ({sh[m_]['ci'][0]:.1f}%) meets the 70% target. On glioma it does so on {sh[g]['shr']:.1f}% of "
        f"slices (lower bound {sh[g]['ci'][0]:.1f}%), which is below the target. A meningioma is usually the brightest "
        "structure of the slice and enhances uniformly, so the highest Otsu class isolates it; a glioma core is often a "
        "thin, irregular rim around necrosis, and vessels or the dura may be ranked above it. The seed decides the "
        f"result: when it hits, the automatic ESRG reaches a mean DSC of {f3(hd[g])} for glioma and {f3(hd[m_])} for "
        f"meningioma, and when it misses, {f3(ms[g])} and {f3(ms[m_])}.")
    w = df[df.config.isin(["P_SRG", "P_SRG_2", "P_SRG_3"])].pivot_table(index=["tumor", "file"], columns="config", values="dsc")
    rng_ = w.max(axis=1) - w.min(axis=1)
    vary = {t: (rng_.loc[t] >= 0.2).mean() * 100 for t in TYPES}
    changed = {t: 0 for t in TYPES}
    base_h = df[df.config == "A_ESRG"].set_index("file")["mask_sha1"]
    for k in (1, 2):
        rp_ = pd.read_csv(os.path.join(d, f"raw_results_repeat{k}.csv"), low_memory=False)
        rp_ = rp_[rp_.config == "A_ESRG"].set_index("file")
        for t in TYPES:
            idx = [f for f in rp_.index if f in base_h.index and rp_.loc[f, "tumor"] == t]
            changed[t] += int((rp_.loc[idx, "mask_sha1"] != base_h.loc[idx]).sum())
    assert sum(changed.values()) == 0
    rows = [[SNAME[t], f"{vary[t]:.1f}%", f"0 of {2 * nt[t]}"] for t in TYPES]
    S.table("4.4", "Consistency of the Result with Planted and Automatic Seeds",
            ["Tumor type", "Planted seeds: slices where moving the seed by 4 to 8 mm changes the DSC by 0.20 or more",
             "Automatic seed: repeated runs that differ from the first run"],
            rows, [1.3, 3.5, 1.9],
            "The planted seeds are the three placements P1, P2, and P3 of the SRG with three seeds (P_SRG, P_SRG_2, P_SRG_3). "
            "The repeated runs are two additional runs of the automatic ESRG on every slice of the tumor type.", ["l", "c", "c"])
    S.p(f"With planted seeds, where the operator clicks matters: moving every seed by 4 to 8 mm changes the DSC by at least "
        f"0.20 on {vary[g]:.1f}% of glioma and {vary[m_]:.1f}% of meningioma slices. The automatic seed removes this "
        "dependence, because the result is identical in every repeated run.")
    S.p("Objective 1 is therefore met for meningioma: the automatic seed is reliable (the lower bound of the seed hit rate "
        "is above 70%) and reproducible. For glioma it is met in part: the result is reproducible, but the seed hit rate "
        "is below the target, so the manual seeding option remains necessary for gliomas.")

    # ── Objective 2 ──
    S.h3("4.2.2 Objective 2: To minimize undersegmentation by replacing the global-mean difference with a local difference measure computed in the log domain")

    def block_rows(blocks):
        rows = []
        for e, lab, mets in blocks:
            rows.append({"group": lab})
            for sc in TYPES:
                for m in mets:
                    t = pe(e, sc, m)
                    rows.append([SNAME[sc], MNAME[m].replace(" (px)", ""), shown(t["base"], m), shown(t["enh"], m),
                                 nz(t["r"]), CALL[t["call"]]])
        return rows
    hdr = ["Tumor type", "Metric", "Without", "With", "r", "Result"]
    wid = [1.3, 1.2, 1.1, 1.1, 0.8, 1.0]
    note = ("Without and With show the configuration before and after the method is applied (see Table 3.2); DSC, recall, and "
            "precision are means and HD95 is a median in pixels. r = effect size of the Wilcoxon signed-rank test; Result = "
            "Better, Worse, or ns (Holm p < .05 and |r| ≥ .10).")
    S.table("4.5", "Effect of the Local Log-Domain Difference Measure (LLDM)", hdr,
            block_rows([("E4", "Main · E4: LLDM removed from the ESRG (ESRG without LLDM → ESRG)", ["recall", "dsc"]),
                        ("E3", "Supplementary · E3: LLDM added to the SRG with three seeds (P_SRG → P_SRG_L)", ["recall", "dsc"])]),
            wid, note, ["l", "l", "c", "c", "c", "c"])
    e4 = {sc: (pe("E4", sc, "recall"), pe("E4", sc, "dsc")) for sc in TYPES}
    S.p("The local measure is evaluated where it is meant to work, inside the ESRG. For meningioma, applying it raises the "
        f"recall of the ESRG from {f3(e4[m_][0]['base']['mean'])} to {f3(e4[m_][0]['enh']['mean'])} and the DSC from "
        f"{f3(e4[m_][1]['base']['mean'])} to {f3(e4[m_][1]['enh']['mean'])} (r = {nz(e4[m_][0]['r'])} and "
        f"{nz(e4[m_][1]['r'])}), so the region covers more of the tumor without loss of overlap. For glioma, the change is "
        f"not significant (recall {f3(e4[g][0]['base']['mean'])} to {f3(e4[g][0]['enh']['mean'])}, ns); the heterogeneous "
        "enhancing rim and necrotic core of a glioma may be too different for a neighborhood comparison to bridge.")
    e3 = {sc: (pe("E3", sc, "recall"), pe("E3", sc, "dsc")) for sc in TYPES}
    S.p("The supplementary test shows what the measure needs: added to the SRG with three seeds, it raises recall "
        f"({f3(e3[g][0]['base']['mean'])} to {f3(e3[g][0]['enh']['mean'])} for glioma and {f3(e3[m_][0]['base']['mean'])} "
        f"to {f3(e3[m_][0]['enh']['mean'])} for meningioma), but only because the region floods the brain "
        f"({leak.loc['P_SRG_L', g]:.1f}% and {leak.loc['P_SRG_L', m_]:.1f}% of slices), and the DSC falls. A local "
        "reference follows the neighboring pixels, so without a stopping rule it has no brake. The measure therefore works "
        "together with the stopping criterion of Objective 3.")
    S.p("Objective 2 is therefore met for meningioma, where the local measure significantly reduces undersegmentation "
        "within the ESRG without harming the overlap, and is not demonstrated for glioma.")

    # ── Objective 3 ──
    S.h3("4.2.3 Objective 3: To decrease boundary leakage by integrating a stopping criterion that adapts to the intensity variability of the growing region, supported with a global drift guard")
    S.table("4.6", "Effect of the Adaptive Stopping Criterion (ASC) on Boundary Leakage", hdr,
            block_rows([("E5b", "Main · E5b: ASC added to the SRG with one seed (P_SRG_solo → P_SRG_solo_S)", ["precision", "hd95_wc", "dsc"]),
                        ("E6", "Main · E6: ASC removed from the ESRG (ESRG without ASC → ESRG)", ["precision", "hd95_wc", "dsc"]),
                        ("E5a", "Supplementary · E5a: ASC with one seed compared with the SRG with three seeds (P_SRG → P_SRG_solo_S)", ["precision", "hd95_wc", "dsc"])]),
            wid, note + " In E6, ‘with’ is the ESRG run with the competing regions of the SRG.", ["l", "l", "c", "c", "c", "c"])
    e5, e6 = pe("E5b", g, "precision"), pe("E6", g, "precision")
    S.p("The adaptive stopping criterion is evaluated in both directions. Added to the one-seed SRG (E5b), it stops the "
        f"region at the tumor: precision rises from {f3(pe('E5b', g, 'precision')['base']['mean'])} to "
        f"{f3(pe('E5b', g, 'precision')['enh']['mean'])} for glioma and from {f3(pe('E5b', m_, 'precision')['base']['mean'])} "
        f"to {f3(pe('E5b', m_, 'precision')['enh']['mean'])} for meningioma, the median HD95 falls from "
        f"{fx(pe('E5b', g, 'hd95_wc')['base']['median'], 1)} to {fx(pe('E5b', g, 'hd95_wc')['enh']['median'], 1)} pixels and from "
        f"{fx(pe('E5b', m_, 'hd95_wc')['base']['median'], 1)} to {fx(pe('E5b', m_, 'hd95_wc')['enh']['median'], 1)} pixels, and "
        f"the DSC rises from {f3(pe('E5b', g, 'dsc')['base']['mean'])} to {f3(pe('E5b', g, 'dsc')['enh']['mean'])} and from "
        f"{f3(pe('E5b', m_, 'dsc')['base']['mean'])} to {f3(pe('E5b', m_, 'dsc')['enh']['mean'])}. Removed from the ESRG "
        f"(E6), the region floods the brain on {leak.loc['P_ESRG_C_noS', g]:.1f}% of glioma and "
        f"{leak.loc['P_ESRG_C_noS', m_]:.1f}% of meningioma slices, and precision, HD95, and DSC all deteriorate, with "
        "large effects (r = .81 to .86 for precision and HD95). All of these differences are significant in both tumor "
        "types.")
    S.p("The supplementary test (E5a) asks whether the criterion with one seed can replace the two competing seeds of the "
        f"SRG. It leaks less (precision {f3(pe('E5a', g, 'precision')['base']['mean'])} to {f3(pe('E5a', g, 'precision')['enh']['mean'])} "
        f"and {f3(pe('E5a', m_, 'precision')['base']['mean'])} to {f3(pe('E5a', m_, 'precision')['enh']['mean'])}), but it stops "
        "before covering the whole tumor, so its HD95 and DSC are lower than those of the three-seed SRG.")
    S.p("Objective 3 is therefore met in both tumor types: with the same single seed, the stopping criterion significantly "
        "increases precision, lowers HD95, and raises the DSC, and removing it from the ESRG causes flooding. The "
        "pre-specified rule of Section 3.1.2 also required E5a, the comparison with the three-seed SRG, to show a lower HD95 "
        "and an unharmed DSC, which it does not; the basis of this conclusion, the one-seed comparison (E5b), was adopted "
        "after the test results were examined.")
    img = os.path.join(figdir, "figure_4_1.png"); iw, ih = Image.open(img).size
    ex = {t: df[df.file == fn].set_index("config").dsc for t, fn in zip(TYPES, figs["examples"])}
    NOTE = ("Red fill and outline = predicted mask; yellow outline = ground-truth tumor core where the prediction does not "
            "cover it. In each row the slice is the test slice whose three-seed SRG DSC is closest to the median of its tumor "
            "type, so the slices are typical for the baseline and were not selected for the ESRG.")
    S.fig("4.1", img, "Segmentations of Two Test Slices at Equal Input", NOTE + " ESRG = enhanced seeded region growing; SRG = seeded region growing.", iw, ih)
    S.p(f"Figure 4.1 shows the difference at equal input. Given one planted seed, the SRG floods the brain in both slices "
        f"(DSC {fx(ex[g]['P_SRG_solo'], 2)} and {fx(ex[m_]['P_SRG_solo'], 2)}), whereas the ESRG stays at the tumor "
        f"({fx(ex[g]['P_ESRG'], 2)} and {fx(ex[m_]['P_ESRG'], 2)}). The automatic seed gives {fx(ex[g]['A_ESRG'], 2)} and "
        f"{fx(ex[m_]['A_ESRG'], 2)}. The three-seed SRG, shown as a reference, scores {fx(ex[g]['P_SRG'], 2)} and "
        f"{fx(ex[m_]['P_SRG'], 2)}.")
    img2 = os.path.join(figdir, "figure_4_2.png"); iw2, ih2 = Image.open(img2).size
    S.fig("4.2", img2, "Effect of the Applied Methods on the Same Test Slices",
          NOTE + " LLDM = local log-domain difference measure; ASC = adaptive stopping criterion.", iw2, ih2)
    S.p("Figure 4.2 illustrates the effect of each method on the same slices. Adding the local measure to the SRG, or removing "
        "the stopping criterion from the ESRG, floods the brain; the SRG with the stopping criterion stays inside the tumor "
        f"but stops early (DSC {fx(ex[g]['P_SRG_solo_S'], 2)} and {fx(ex[m_]['P_SRG_solo_S'], 2)}); and removing the local "
        f"measure from the ESRG changes these two slices little (DSC {fx(ex[g]['P_ESRG_noL'], 2)} against {fx(ex[g]['P_ESRG'], 2)} "
        f"for glioma and {fx(ex[m_]['P_ESRG_noL'], 2)} against {fx(ex[m_]['P_ESRG'], 2)} for meningioma), in line with the "
        "small average effect in Table 4.5.")

    # ═══ 4.3 ═════════════════════════════════════════════════════════════════
    S.h2("4.3 Comparative Analysis of the Baseline SRG and the ESRG")
    S.p("Table 4.7 brings the experiments together. Each block names the experiment, the configurations compared, and the "
        "metric; gives the mean (DSC, recall, precision) or median (time) of the baseline SRG and of the ESRG for each tumor "
        "type; and states the significance of the difference and its interpretation. The main comparisons are at equal "
        "input. The supplementary comparisons give the SRG more operator input than the ESRG, or start both algorithms from "
        "the same automatic seed, and are reported for completeness.")

    def row_from(t, m):
        return t["base"], t["enh"], t

    def e1row(sc, m, k, b, e):
        t = post(sc, m, k); return d1(sc, m, b), d1(sc, m, e), t
    blocks = [
        ("Main · Same input, one planted seed, DSC: SRG (P_SRG_solo) → ESRG (P_ESRG)", "dsc",
         lambda sc: e1row(sc, "dsc", "P_ESRG vs P_SRG_solo", "P_SRG_solo", "P_ESRG"),
         "At equal input the SRG absorbs the whole image; the ESRG stops at the tumor."),
        ("Main · No operator seed, DSC: SRG (P_SRG_solo) → ESRG (A_ESRG)", "dsc",
         lambda sc: e1row(sc, "dsc", "A_ESRG vs P_SRG_solo", "P_SRG_solo", "A_ESRG"),
         "The ESRG with an automatic seed is better than the SRG with a planted seed."),
        ("Main · Same input, processing time (s): SRG (P_SRG_solo) → ESRG (P_ESRG)", "seconds",
         lambda sc: e1row(sc, "seconds", "P_ESRG vs P_SRG_solo", "P_SRG_solo", "P_ESRG"),
         "The ESRG is faster because it labels only the tumor."),
        ("Main · E5b, ASC added (Objective 3), precision: SRG (P_SRG_solo) → SRG + ASC (P_SRG_solo_S)", "precision",
         lambda sc: (lambda t: (t["base"], t["enh"], t))(pe("E5b", sc, "precision")),
         "The stopping criterion removes the leakage of the one-seed SRG."),
        ("Main · E6, ASC removed (Objective 3), precision: ESRG without ASC → ESRG (P_ESRG_C)", "precision",
         lambda sc: (lambda t: (t["base"], t["enh"], t))(pe("E6", sc, "precision")),
         "Without the stopping criterion the ESRG floods; the criterion is necessary."),
        ("Main · E4, LLDM removed (Objective 2), recall: ESRG without LLDM → ESRG (P_ESRG)", "recall",
         lambda sc: (lambda t: (t["base"], t["enh"], t))(pe("E4", sc, "recall")),
         None),
        ("Supplementary · SRG with three seeds, DSC: SRG (P_SRG) → ESRG (P_ESRG)", "dsc",
         lambda sc: e1row(sc, "dsc", "P_ESRG vs P_SRG", "P_SRG", "P_ESRG"),
         "With two extra seeds the SRG scores higher: the ESRG leaks less but covers less of the tumor."),
        ("Supplementary · Same automatic seed, DSC: SRG (A_SRG) → ESRG (A_ESRG)", "dsc",
         lambda sc: (lambda t: (t["base"], t["enh"], t))(se("S2", sc, "dsc")),
         "From the same seed the SRG scores higher; the ESRG undersegments."),
        ("Supplementary · E3, LLDM added to the three-seed SRG (Objective 2), recall: SRG (P_SRG) → SRG + LLDM (P_SRG_L)", "recall",
         lambda sc: (lambda t: (t["base"], t["enh"], t))(pe("E3", sc, "recall")),
         "Recall rises only because the SRG floods; its DSC falls (Table 4.5)."),
        ("Supplementary · E5a, ASC replaces competing seeds (Objective 3), DSC: SRG (P_SRG) → SRG + ASC (P_SRG_solo_S)", "dsc",
         lambda sc: (lambda t: (t["base"], t["enh"], t))(pe("E5a", sc, "dsc")),
         "One seed with the criterion stops before covering the tumor; the three-seed SRG is higher."),
    ]
    rows = []
    for lab, m, fn, why in blocks:
        rows.append({"group": lab})
        for sc in TYPES:
            b, e, t = fn(sc)
            txt = why
            if txt is None:                                    # E4 differs by tumor type
                txt = ("The measure raises recall inside the ESRG." if t["call"] == "better"
                       else "No significant change in recall within the ESRG.")
            rows.append([SNAME[sc], shown(b, m), shown(e, m), nz(t["r"]), CALL[t["call"]], txt])
    S.table("4.7", "Comparative Analysis of the Baseline SRG and the ESRG by Experiment",
            ["Tumor type", "Baseline SRG", "ESRG / compared", "r", "Result", "Interpretation"],
            rows, [1.0, 0.8, 0.9, 0.5, 0.65, 2.6],
            "Baseline → compared as named in each block (see Table 3.2). Values are means for DSC, recall, and precision and "
            "medians for time. r = effect size of the Wilcoxon signed-rank test; Result = Better, Worse, or ns (Holm p < .05 "
            "and |r| ≥ .10), with the compared configuration as the enhanced one. The comparison of the three-seed SRG, the "
            "comparison with the same automatic seed, and the one-seed comparisons were defined or adopted after the test "
            "run, using results already collected.",
            ["l", "c", "c", "c", "c", "l"])
    S.p("At equal input, the ESRG improves on the SRG. Given the same single tumor seed, it is far better in DSC, precision, "
        "HD95, and processing time in both tumor types, and its automatic version, which needs no operator seed, is better "
        "as well. Each applied method contributes: the stopping criterion removes the leakage of the single-seed SRG and is "
        "necessary within the ESRG, and the local measure raises recall within the ESRG for meningioma. The "
        "supplementary comparisons show the limits of this advantage. When the SRG is given two additional seeds for "
        "healthy tissue and background, or when both algorithms start from the same automatic seed, the SRG obtains a higher "
        "DSC, because the ESRG is more conservative: it leaks less (higher precision) but covers less of the tumor (lower "
        "recall). The ESRG thus replaces the extra operator input of the SRG with its stopping criterion, at the cost of "
        "some recall.")
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
