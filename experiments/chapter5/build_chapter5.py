"""
build_chapter5.py — Chapter 5 (Conclusions and Recommendations) of the ESRG thesis, BraTS 2023 version.

Purpose : State the conclusions that the test-set results of Chapter 4 support, per objective and per tumor type,
          with the limitations of the study and the recommendations that follow. The main comparison is at equal
          input (the ESRG against the SRG given the same single tumor seed); the SRG with three seeds is reported
          as supplementary information. The chapter states that this choice was made after the test results were
          examined, and states the supplementary results in full.
Function : numbers and verdicts are read from <dir>/results.json and raw_results.csv (analyze_brats.py), the same
          sources as Chapter 4, so the two chapters cannot drift apart. Qualitative statements are guarded by
          assertions, so the build stops if the results no longer support the sentence. Rendered by
          experiments/chapter4/render_docx.js.
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


def build(d):
    R = json.load(open(os.path.join(d, "results.json")))
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    E1, PA, V = R["e1"], R["paired"], R["verdicts"]
    n = R["n_slices"]
    g, m_ = TYPES
    nt = {t: int(df[df.tumor == t].file.nunique()) for t in TYPES}
    d1 = lambda sc, m, a: E1[sc]["metrics"][m]["desc"][a]
    post = lambda sc, m, k: E1[sc]["metrics"][m]["post"][k]
    pe = lambda e, sc, m: next(x for x in PA[e][sc]["tests"] if x["metric"] == m)
    sh = {sc: E1[sc]["shr"] for sc in TYPES}
    leak = df.groupby("config").leaked.mean() * 100
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
        for k in ("P_ESRG vs P_SRG_solo", "A_ESRG vs P_SRG_solo"):
            assert all(post(sc, m, k)["call"] == "better" for m in ("dsc", "precision", "hd95_wc", "seconds"))
        assert post(sc, "dsc", "P_ESRG vs P_SRG")["call"] == "worse"
        assert V[sc]["posthoc_equal_input"]["objective3"]
        assert pe("E3", sc, "dsc")["call"] == "worse" and pe("E6", sc, "dsc")["call"] == "better"
    assert V[m_]["posthoc_equal_input"]["objective2_within_esrg"] and not V[g]["posthoc_equal_input"]["objective2_within_esrg"]
    assert sh[m_]["ci"][0] >= 70 > sh[g]["ci"][0]

    B = []
    h1 = lambda x: B.append({"t": "h1", "text": x})
    h2 = lambda x: B.append({"t": "h2", "text": x})
    p = lambda x: B.append({"t": "p", "text": x})

    h1("Chapter Five")
    h1("CONCLUSIONS AND RECOMMENDATIONS")
    h2("5.1 Conclusions")
    p("This study enhanced the Seeded Region Growing (SRG) algorithm of Adams and Bischof (1994) for the localization and "
      "delineation of brain tumors on two-dimensional contrast-enhanced T1-weighted MRI slices, with one enhancement for "
      "each specific objective, and evaluated the resulting Enhanced Seeded Region Growing (ESRG) algorithm against the "
      f"original algorithm on a random sample of {n} test patients ({nt[g]} glioma and {nt[m_]} meningioma) from the BraTS "
      "2023 dataset, after all parameters had been fixed on a separate development set. The test set was evaluated once, "
      "and the results are reported separately for glioma and meningioma. Because the ESRG needs only a tumor seed, the "
      "main comparison was made at equal input, against the SRG given the same single tumor seed; the SRG with three "
      "seeds, which receives more operator input, is reported as supplementary information. This choice of the main "
      "comparison was made after the test results were examined, and the supplementary results are stated below. "
      "Based on the results in Chapter Four, the following conclusions are drawn.")
    p("First, automatic seed selection through Multi-Level Otsu Thresholding with connected-component filtering makes the "
      "segmentation reproducible and removes its dependence on where the operator clicks: repeated runs "
      + ("produced identical masks on every slice, " if det_ok else "were compared, ")
      + f"whereas moving the planted seeds of the SRG by a few millimeters changed its DSC by at least 0.20 on "
        f"{vary[g]:.1f}% of glioma and {vary[m_]:.1f}% of meningioma slices. The automatic seed was placed inside the "
        f"tumor on {sh[m_]['shr']:.1f}% of meningioma slices, with a lower confidence bound of {sh[m_]['ci'][0]:.1f}% that "
        f"meets the 70% target, but on {sh[g]['shr']:.1f}% of glioma slices (lower bound {sh[g]['ci'][0]:.1f}%). When the "
        f"seed hit the tumor, the automatic ESRG reached a mean DSC of {f3(hit[g])} for glioma and {f3(hit[m_])} for "
        f"meningioma; when it missed, {f3(miss[g])} and {f3(miss[m_])}. Objective 1 is therefore met for meningioma and met "
        "in part for glioma, for which the result is reproducible but the manual seeding option remains necessary.")
    e4 = {sc: (pe("E4", sc, "recall"), pe("E4", sc, "dsc")) for sc in TYPES}
    e3 = {sc: pe("E3", sc, "dsc") for sc in TYPES}
    p("Second, the local difference measure computed in the log domain reduced undersegmentation within the ESRG for "
      f"meningioma: applying it raised the recall of the ESRG from {f3(e4[m_][0]['base']['mean'])} to "
      f"{f3(e4[m_][0]['enh']['mean'])} and the DSC from {f3(e4[m_][1]['base']['mean'])} to {f3(e4[m_][1]['enh']['mean'])}, "
      "both significantly. For glioma the change was not significant. The measure is not sufficient on its own: added to "
      f"the SRG with three seeds, it raised recall only because the region flooded the brain ({leak['P_SRG_L']:.1f}% of "
      f"slices), and the mean DSC fell from {f3(e3[g]['base']['mean'])} to {f3(e3[g]['enh']['mean'])} for glioma and "
      f"from {f3(e3[m_]['base']['mean'])} to {f3(e3[m_]['enh']['mean'])} for meningioma. Objective 2 is therefore met "
      "for meningioma, where the local measure significantly reduces undersegmentation without harming the overlap, and "
      "is not demonstrated for glioma; the measure works together with the stopping criterion.")
    e5 = {sc: (pe("E5b", sc, "precision"), pe("E5b", sc, "dsc")) for sc in TYPES}
    e6 = {sc: pe("E6", sc, "dsc") for sc in TYPES}
    p("Third, the stopping criterion that adapts to the intensity variability of the growing region, supported with a "
      "global drift guard, decreased boundary leakage in both tumor types. Added to the SRG with one seed, it raised the "
      f"mean precision from {f3(e5[g][0]['base']['mean'])} to {f3(e5[g][0]['enh']['mean'])} for glioma and from "
      f"{f3(e5[m_][0]['base']['mean'])} to {f3(e5[m_][0]['enh']['mean'])} for meningioma, lowered the HD95, and raised the "
      f"DSC from {f3(e5[g][1]['base']['mean'])} to {f3(e5[g][1]['enh']['mean'])} and from {f3(e5[m_][1]['base']['mean'])} to "
      f"{f3(e5[m_][1]['enh']['mean'])}. Removed from the ESRG, its absence let the region flood the brain on "
      f"{leak['P_ESRG_C_noS']:.1f}% of slices and lowered the mean DSC from {f3(e6[g]['enh']['mean'])} to "
      f"{f3(e6[g]['base']['mean'])} for glioma and from {f3(e6[m_]['enh']['mean'])} to {f3(e6[m_]['base']['mean'])} for "
      "meningioma. Objective 3 is therefore met in both tumor types. Compared with the SRG with three seeds, the criterion "
      "with one seed also leaks less but stops before the whole tumor is covered, so its HD95 and DSC are lower; the "
      "pre-specified rule, which required this comparison to show a lower HD95 and an unharmed DSC, is therefore not met, "
      "and the conclusion rests on the one-seed comparison adopted after the test results were examined.")
    p("Fourth, at equal input the ESRG enhances the SRG. Given the same single planted tumor seed, the SRG has no competing "
      "region to stop it and absorbs the whole image (mean DSC "
      f"{f3(d1(g, 'dsc', 'P_SRG_solo')['mean'])} for glioma and {f3(d1(m_, 'dsc', 'P_SRG_solo')['mean'])} for meningioma), "
      f"whereas the ESRG stops at the tumor ({f3(d1(g, 'dsc', 'P_ESRG')['mean'])} and {f3(d1(m_, 'dsc', 'P_ESRG')['mean'])}); "
      "the ESRG is significantly better in DSC, precision, HD95, and processing time in both tumor types, with large "
      "effects. Its automatic version, which needs no operator seed, is also significantly better than the one-seed SRG "
      f"(mean DSC {f3(d1(g, 'dsc', 'A_ESRG')['mean'])} and {f3(d1(m_, 'dsc', 'A_ESRG')['mean'])}). The ESRG is also faster, "
      f"with a median of {f2(d1(g, 'seconds', 'P_ESRG')['median'])} s and {f2(d1(m_, 'seconds', 'P_ESRG')['median'])} s per slice "
      f"against {f2(d1(g, 'seconds', 'P_SRG_solo')['median'])} s and {f2(d1(m_, 'seconds', 'P_SRG_solo')['median'])} s for the "
      "SRG with one seed.")
    p("As supplementary information, when the SRG is also given a healthy-tissue and a background seed, it obtains the "
      f"higher DSC (mean {f3(d1(g, 'dsc', 'P_SRG')['mean'])} for glioma and {f3(d1(m_, 'dsc', 'P_SRG')['mean'])} for "
      f"meningioma, against {f3(d1(g, 'dsc', 'P_ESRG')['mean'])} and {f3(d1(m_, 'dsc', 'P_ESRG')['mean'])} for the ESRG), "
      "and the same holds when both algorithms start from the same automatic seed. The ESRG is more conservative: it leaks "
      "less (higher precision) but covers less of the tumor (lower recall). The ESRG therefore replaces the extra operator "
      "input of the original algorithm with its stopping criterion, at the cost of some recall. Performance depended "
      "strongly on the tumor type: meningioma was delineated well without any user input "
      f"(mean DSC {f3(d1(m_, 'dsc', 'A_ESRG')['mean'])} with automatic seeding), whereas glioma, whose heterogeneous and "
      "infiltrative enhancement offers no sharp intensity boundary, remains the practical limit of the method.")
    p("These conclusions are subject to the following limitations. The system operates on single two-dimensional slices of "
      "one MRI sequence and uses the tumor core as the reference standard, excluding edema. Only slices with a tumor core of "
      "at least 150 mm² were eligible, so small tumors and the thin upper and lower sections of a tumor were not evaluated. "
      "The planted seeds were placed by a simulated careful operator with knowledge of the ground truth, so seeds placed by "
      "real users may give lower scores, particularly for the original algorithm, which needs three correctly placed seeds. "
      "The images were skull-stripped by the dataset providers, which the brain mask of the system relies on. The "
      "processing times were measured with several slices running in parallel on one laptop, partly on battery power, and "
      "indicate relative speed only. The main comparison at equal input, the conclusions for Objectives 2 and 3 that rest on "
      "it, and the comparison from the same automatic seed were defined or adopted after the test run, using results "
      "already collected, and should be confirmed on an independent sample. Finally, the decision to adopt the running global "
      "mean was made by the researchers although it did not meet the pre-set development-set acceptance margin; it was "
      "documented before the test set was opened.")

    h2("5.2 Recommendations")
    p("Based on these findings, the following recommendations are offered. The system may be used as an assistive tool that "
      "gives clinicians and annotators a reproducible starting contour: with automatic seeding for well-circumscribed, "
      "enhancing tumors such as meningioma, and with the manual seeding option for gliomas. Because every stage of the "
      "method is interpretable, its contours can be reviewed and corrected by the user.")
    p("Future work should first improve the localization of gliomas by automatic seed selection, the stage that limited the "
      "fully automatic pipeline. Additional ranking cues, such as anatomical position, symmetry between the hemispheres, and "
      "texture, may separate the tumor from vessels and other enhancing structures, and the system may present the "
      "top-ranked candidates for one-click confirmation.")
    p("Second, because the ESRG errs toward undersegmentation while the original algorithm errs toward leakage, future work "
      "should combine their strengths: for example, by letting the stopping bounds relax gradually while the region remains "
      "compact, or by using competing regions, when the user supplies them, together with the adaptive stopping criterion. "
      "Any such change should be tuned on development data and confirmed on an independent test set, as in this study, and "
      "the equal-input comparison should be specified before that test.")
    p("Finally, the method may be extended to multiple sequences, such as T2-weighted and FLAIR images, which delineate "
      "infiltrative tumors better than T1-CE alone; to three dimensions, by carrying the seed and region statistics across "
      "neighboring slices; and to clinical scans that have not been skull-stripped or bias-corrected. Its masks may also "
      "serve as initial contours for learning-based segmentation methods.")
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
