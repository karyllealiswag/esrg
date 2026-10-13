"""
build_chapter5.py — Chapter 5 (Conclusions and Recommendations) of the ESRG thesis, BraTS 2023 version.

Purpose : State the conclusions that the test-set results of Chapter 4 support, per objective and per tumor
          type, with the limitations of the study and the recommendations that follow from the findings.
Function : numbers and verdicts are read from <dir>/results.json and raw_results.csv (analyze_brats.py), the
          same sources as Chapter 4, so the two chapters cannot drift apart; results are given for glioma and
          meningioma separately. Qualitative statements are guarded by assertions, so the build stops if the
          results no longer support the sentence. Rendered by experiments/chapter4/render_docx.js.
          CLI: python -P experiments/chapter5/build_chapter5.py [--dir outputs/evaluation_brats/test]
"""
import argparse
import json
import os
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TYPES = ["glioma", "meningioma"]


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def both(fn, unit=""):
    """'a for glioma and b for meningioma' (the pair is written out so the sentence names each tumor type)."""
    return f"{fn(TYPES[0])}{unit} for glioma and {fn(TYPES[1])}{unit} for meningioma"


def build(d):
    R = json.load(open(os.path.join(d, "results.json")))
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    E1, E2, PA, V = R["e1"], R["e2"], R["paired"], R["verdicts"]
    n = R["n_slices"]
    g, m_ = TYPES
    nt = {t: int(df[df.tumor == t].file.nunique()) for t in TYPES}
    e1d = lambda sc, m, a: E1[sc]["metrics"][m]["desc"][a]
    post = lambda sc, m, k: E1[sc]["metrics"][m]["post"][k]
    test = lambda e, sc, m: next(x for x in PA[e][sc]["tests"] if x["metric"] == m)
    sh = {sc: E1[sc]["shr"] for sc in TYPES}
    leak = df.groupby(["config", "tumor"]).leaked.mean().unstack() * 100
    a = df[df.config == "A_ESRG"].copy()
    a["hit"] = a.seed_hit.fillna(False).astype(bool)
    hit = {t: a[(a.tumor == t) & a.hit].dsc.mean() for t in TYPES}
    miss = {t: a[(a.tumor == t) & ~a.hit].dsc.fillna(0).mean() for t in TYPES}
    w = df[df.config.isin(["P_SRG", "P_SRG_2", "P_SRG_3"])].pivot_table(index=["tumor", "file"], columns="config", values="dsc")
    rng_ = w.max(axis=1) - w.min(axis=1)
    vary = {t: (rng_.loc[t] >= 0.2).mean() * 100 for t in TYPES}
    det = R["determinism"]
    det_ok = all(v["identical"] == v["n"] for v in det.values()) and len(det) == 4

    # ── claims the text relies on ──
    for sc in TYPES:
        assert post(sc, "dsc", "P_ESRG vs P_SRG_solo")["call"] == "better"
        assert post(sc, "dsc", "P_ESRG vs P_SRG")["call"] == "worse"
        assert post(sc, "precision", "P_ESRG vs P_SRG")["call"] == "better"
        assert post(sc, "recall", "P_ESRG vs P_SRG")["call"] == "worse"
        assert post(sc, "seconds", "P_ESRG vs P_SRG")["call"] == "better"
        assert test("E3", sc, "recall")["call"] == "better" and test("E3", sc, "dsc")["call"] == "worse"
        assert test("E6", sc, "precision")["call"] == "better" and test("E6", sc, "dsc")["call"] == "better"
        assert test("E5a", sc, "precision")["call"] == "better" and test("E5a", sc, "dsc")["call"] == "worse"
        assert not V[sc]["overall"] and not V[sc]["objective2"] and not V[sc]["objective3"]
        assert not E2["E2a"][sc]["auto_not_lower"]
    assert test("E4", m_, "recall")["call"] == "better" and test("E4", m_, "dsc")["call"] == "better"
    assert test("E4", g, "recall")["call"] == "no effect" and test("E4", g, "dsc")["call"] == "no effect"
    assert sh[m_]["ci"][0] >= 70 > sh[g]["ci"][0]

    B = []
    h1 = lambda x: B.append({"t": "h1", "text": x})
    h2 = lambda x: B.append({"t": "h2", "text": x})
    p = lambda x: B.append({"t": "p", "text": x})

    h1("Chapter Five")
    h1("CONCLUSIONS AND RECOMMENDATIONS")
    h2("5.1 Conclusions")
    p("This study enhanced the Seeded Region Growing (SRG) algorithm of Adams and Bischof (1994) for the localization "
      "and delineation of brain tumors on two-dimensional contrast-enhanced T1-weighted MRI slices, with one "
      "enhancement for each specific objective, and evaluated the resulting Enhanced Seeded Region Growing (ESRG) "
      f"algorithm against the original algorithm on a random sample of {n} test patients ({nt[g]} glioma and "
      f"{nt[m_]} meningioma) from the BraTS 2023 dataset, after all parameters had been fixed on a separate "
      "development set. Both algorithms received the same slices, seeds, and parameter values, and the test set was "
      "evaluated once. The results are reported separately for glioma and meningioma. Based on the results in "
      "Chapter Four, the following conclusions are drawn.")

    # Objective 1
    p("First, automatic seed selection through Multi-Level Otsu Thresholding with connected-component filtering makes "
      "the segmentation reproducible and removes its dependence on where the operator clicks: repeated runs "
      + ("produced identical masks on every slice, " if det_ok else "were compared, ")
      + f"whereas moving the planted seeds of the SRG by a few millimeters changed its DSC by at least 0.20 on "
        f"{vary[g]:.1f}% of glioma and {vary[m_]:.1f}% of meningioma slices. The automatic seed was placed inside the "
        f"tumor on {sh[m_]['shr']:.1f}% of meningioma slices, with a lower confidence bound of {sh[m_]['ci'][0]:.1f}% "
        f"that meets the 70% target, but only on {sh[g]['shr']:.1f}% of glioma slices (lower bound {sh[g]['ci'][0]:.1f}%). "
        f"When the seed hit the tumor, the automatic ESRG reached a mean DSC of {f3(hit[g])} for glioma and {f3(hit[m_])} "
        f"for meningioma; when it missed, {f3(miss[g])} and {f3(miss[m_])}. The automatic DSC was nevertheless "
        "significantly lower than that of the planted placements in both tumor types. Objective 1 is therefore "
        "achieved for meningioma but not for glioma, for which the manual seeding option remains necessary.")

    # Objective 2
    e3 = {sc: (test("E3", sc, "recall"), test("E3", sc, "dsc")) for sc in TYPES}
    e4 = {sc: (test("E4", sc, "recall"), test("E4", sc, "dsc")) for sc in TYPES}
    p("Second, the local difference measure computed in the log domain did not, by itself, minimize undersegmentation "
      "in a way that improves the segmentation. Added to the original algorithm, it raised the median recall to "
      f"{f3(e3[g][0]['enh']['median'])} in both tumor types, but only because the region flooded the brain on "
      f"{leak.loc['P_SRG_L', g]:.1f}% of glioma and {leak.loc['P_SRG_L', m_]:.1f}% of meningioma slices, and the median "
      f"DSC fell from {f3(e3[g][1]['base']['median'])} to {f3(e3[g][1]['enh']['median'])} and from "
      f"{f3(e3[m_][1]['base']['median'])} to {f3(e3[m_][1]['enh']['median'])}. Within the ESRG, where the stopping "
      f"criterion bounds the region, the measure significantly increased recall (median {f3(e4[m_][0]['base']['median'])} → "
      f"{f3(e4[m_][0]['enh']['median'])}) and DSC for meningioma, but not for glioma. Objective 2 is therefore met only in "
      "part: the measure reduces undersegmentation within the ESRG for meningioma, but it is not sufficient on its own "
      "and is useful only in combination with a stopping criterion.")

    # Objective 3
    e6 = {sc: (test("E6", sc, "precision"), test("E6", sc, "dsc")) for sc in TYPES}
    e5 = {sc: (test("E5a", sc, "precision"), test("E5a", sc, "dsc")) for sc in TYPES}
    p("Third, the stopping criterion that adapts to the intensity variability of the growing region, supported with a "
      "global drift guard, decreased boundary leakage in every comparison. Removed from the ESRG, its absence let the "
      f"region flood the brain on {leak.loc['P_ESRG_C_noS', g]:.1f}% of glioma and {leak.loc['P_ESRG_C_noS', m_]:.1f}% of "
      f"meningioma slices and lowered the median DSC from {f3(e6[g][1]['enh']['median'])} to {f3(e6[g][1]['base']['median'])} "
      f"and from {f3(e6[m_][1]['enh']['median'])} to {f3(e6[m_][1]['base']['median'])}. From the tumor seed alone, it kept the "
      f"original algorithm inside the tumor more strictly than two competing seeds did (median precision "
      f"{f3(e5[g][0]['base']['median'])} → {f3(e5[g][0]['enh']['median'])} for glioma and {f3(e5[m_][0]['base']['median'])} → "
      f"{f3(e5[m_][0]['enh']['median'])} for meningioma), but it also stopped the region before the whole tumor was "
      f"covered, so the DSC was lower (median {f3(e5[g][1]['base']['median'])} → {f3(e5[g][1]['enh']['median'])} and "
      f"{f3(e5[m_][1]['base']['median'])} → {f3(e5[m_][1]['enh']['median'])}). Objective 3 is therefore met with respect to "
      "leakage, measured by precision, but not under the full decision rule, which also required a lower boundary "
      "distance and an unharmed DSC.")

    # Overall comparison
    p("Fourth, the comparison of the two algorithms depends on the input the operator gives. With the same single "
      "planted tumor seed, the original algorithm has no competing region to stop it and absorbs the whole image "
      f"(median DSC {both(lambda t: f3(e1d(t, 'dsc', 'P_SRG_solo')['median']))}), whereas the ESRG stops at the tumor "
      f"({both(lambda t: f3(e1d(t, 'dsc', 'P_ESRG')['median']))}); the difference is large and significant in both tumor "
      "types. When the original algorithm also receives a healthy-tissue and a background seed, it obtains the higher "
      f"DSC (median {both(lambda t: f3(e1d(t, 'dsc', 'P_SRG')['median']))}, against the ESRG’s values above), and the same "
      "holds when both algorithms start from the same automatic seed. The ESRG leaks less (higher precision) and needs "
      "only a single seed, or none, but it recovers less of the tumor (lower recall). It is also much faster, with a "
      f"median processing time of {both(lambda t: f2(e1d(t, 'seconds', 'P_ESRG')['median']), ' s')} with a planted seed "
      f"against {both(lambda t: f2(e1d(t, 'seconds', 'P_SRG')['median']), ' s')} for the original algorithm with three seeds. "
      "The study therefore does not demonstrate an overall improvement in segmentation accuracy over the original "
      "algorithm supplied with competing seeds, but it does show that the ESRG delivers a controlled, fast, and "
      "reproducible segmentation from one seed where the original algorithm cannot.")
    p("Taken together, the enhancements are complementary rather than independent: the local log-domain measure lets "
      "the region reach the tumor boundary, and the stopping criterion keeps it from crossing it. Performance depended "
      "strongly on the tumor type: meningioma was delineated well without any user input "
      f"(median DSC {f3(e1d(m_, 'dsc', 'A_ESRG')['median'])} with automatic seeding), whereas glioma, whose "
      "heterogeneous and infiltrative enhancement offers no sharp intensity boundary, remains the practical limit of "
      f"the method (median DSC {f3(e1d(g, 'dsc', 'A_ESRG')['median'])} with automatic seeding and "
      f"{f3(e1d(g, 'dsc', 'P_ESRG')['median'])} with a planted seed).")
    p("These conclusions are subject to the following limitations. The system operates on single two-dimensional "
      "slices of one MRI sequence and uses the tumor core as the reference standard, excluding edema. Only slices "
      "with a tumor core of at least 150 mm² were eligible, so small tumors and the thin upper and lower sections of a "
      "tumor were not evaluated. The planted seeds were placed by a simulated careful operator with knowledge of the "
      "ground truth, so seeds placed by real users may give lower scores, particularly for the original algorithm, "
      "which needs three correctly placed seeds. The images were skull-stripped by the dataset providers, which the "
      "brain mask of the system relies on. The processing times were measured with several slices running in "
      "parallel on one laptop, partly on battery power, and indicate relative speed only. The comparison at equal "
      "input and the comparison from the same automatic seed were added after the test run, using results already "
      "collected, and are descriptive. Finally, the decision to adopt the running global mean was made by the "
      "researchers although it did not meet the pre-set development-set acceptance margin; it was documented before "
      "the test set was opened.")

    h2("5.2 Recommendations")
    p("Based on these findings, the following recommendations are offered. The system may be used as an assistive "
      "tool that gives clinicians and annotators a reproducible starting contour: with automatic seeding for "
      "well-circumscribed, enhancing tumors such as meningioma, and with the manual seeding option for gliomas. "
      "Because every stage of the method is interpretable, its contours can be reviewed and corrected by the user.")
    p("Future work should first improve the localization of gliomas by automatic seed selection, the stage that "
      "limited the fully automatic pipeline. Additional ranking cues, such as anatomical position, symmetry between "
      "the hemispheres, and texture, may separate the tumor from vessels and other enhancing structures, and the "
      "system may present the top-ranked candidates for one-click confirmation.")
    p("Second, because the ESRG errs toward undersegmentation while the original algorithm errs toward leakage, "
      "future work should combine their strengths: for example, by letting the stopping bounds relax gradually while "
      "the region remains compact, or by using competing regions, when the user supplies them, together with the "
      "adaptive stopping criterion. Any such change should be tuned on development data and confirmed on an "
      "independent test set, as in this study.")
    p("Finally, the method may be extended to multiple sequences, such as T2-weighted and FLAIR images, which "
      "delineate infiltrative tumors better than T1-CE alone; to three dimensions, by carrying the seed and region "
      "statistics across neighboring slices; and to clinical scans that have not been skull-stripped or "
      "bias-corrected. Its masks may also serve as initial contours for learning-based segmentation methods.")
    return {"title": "Chapter 5: Conclusions and Recommendations",
            "author": "Karylle Vinces J. Aliswag & Juana Mari A. Del Rosario", "blocks": B}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "outputs", "evaluation_brats", "test"))
    ap.add_argument("--out", default=os.path.join(ROOT, "Chapter5_Conclusions_and_Recommendations.docx"))
    a = ap.parse_args()
    spec = build(a.dir)
    sp = os.path.join(ROOT, "outputs", "chapter4_brats", "chapter5_spec.json")
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    subprocess.run(["node", os.path.join(ROOT, "experiments", "chapter4", "render_docx.js"), sp, a.out], check=True)
    print("blocks:", len(spec["blocks"]))
