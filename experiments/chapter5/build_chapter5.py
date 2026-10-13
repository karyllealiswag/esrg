"""
build_chapter5.py — Chapter 5 (Conclusions and Recommendations) of the ESRG thesis, BraTS 2023 version.

Purpose : State the conclusions that the test-set results of Chapter 4 support, per objective, with
          the limitations of the study and the recommendations that follow from the findings.
Function : numbers and verdicts are read from <dir>/results.json and raw_results.csv (analyze_brats.py),
          so the conclusions cannot drift from Chapter 4; rendered by experiments/chapter4/render_docx.js.
          CLI: python -P experiments/chapter5/build_chapter5.py [--dir outputs/evaluation_brats/test]
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def f3(x): return f"{x:.3f}"


def build(d):
    R = json.load(open(os.path.join(d, "results.json")))
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    D, E1, V, PA = R["descriptives"], R["e1"], R["verdicts"], R["paired"]
    n = R["n_slices"]
    nt = {t: int(df[df.tumor == t].file.nunique()) for t in ("glioma", "meningioma")}
    m = lambda a, sc, k="mean", met="dsc": D[a][sc][met][k]
    sh = {sc: E1[sc]["shr"] for sc in ("glioma", "meningioma", "pooled")}
    t = lambda e, met: next(x for x in PA[e]["pooled"]["tests"] if x["metric"] == met)
    leak = df.groupby("config").leaked.mean() * 100
    s1 = lambda met: next(x for x in R["secondary"]["S1"]["pooled"]["tests"] if x["metric"] == met)
    overall = V["pooled"]["overall"]
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
      f"algorithm against the original algorithm on a random sample of {n} test patients ({nt['glioma']} glioma and "
      f"{nt['meningioma']} meningioma) from the BraTS 2023 dataset, after all parameters had been fixed on a separate "
      "development set. Both algorithms received the same slices, seeds, brain mask, and parameter values, and the "
      "test set was evaluated once. Based on the results in Chapter Four, the following conclusions are drawn.")
    p(f"First, automatic seed selection through Multi-Level Otsu Thresholding with connected-component filtering "
      f"makes the segmentation fully reproducible and removes its dependence on where the operator clicks; repeated "
      f"runs produced identical masks, whereas moving the planted seeds by a few millimeters changed the result "
      f"substantially on a minority of slices. The automatic seed was placed inside the tumor on {sh['pooled']['shr']:.1f}% "
      f"of slices: on {sh['meningioma']['shr']:.1f}% of meningioma slices, which "
      f"{'meets' if sh['meningioma']['ci'][0] >= 70 else 'does not meet'} the 70% target, but only on "
      f"{sh['glioma']['shr']:.1f}% of glioma slices. When the seed was placed inside the tumor, the automatic ESRG "
      "performed close to the ESRG with a planted seed; when it missed, the segmentation failed. Objective 1 is "
      + ("therefore met." if all(V["pooled"]["objective1"].values()) else
         "therefore achieved for meningioma but not for glioma, for which the manual seeding option remains necessary."))
    e3r, e3d, e4r, e4d = t("E3", "recall"), t("E3", "dsc"), t("E4", "recall"), t("E4", "dsc")
    p("Second, the local difference measure computed in the log domain did not, by itself, minimize "
      "undersegmentation in a way that improves the segmentation. Added to the original algorithm, it raised recall "
      f"(median {f3(e3r['base']['median'])} to {f3(e3r['enh']['median'])}) only because the region flooded the brain on "
      f"{leak.get('P_SRG_L', float('nan')):.0f}% of slices, and the DSC fell sharply. Within the ESRG, where the stopping "
      "criterion bounds the region, the measure "
      + ("significantly increased recall and DSC" if e4r["call"] == "better" and e4d["call"] == "better" else
         "increased recall" if e4r["call"] == "better" else "did not significantly increase recall")
      + " when pooled and for meningioma, but not for glioma. Objective 2 is "
      + ("therefore met." if V["pooled"]["objective2"] else
         "therefore met only in part: the measure reduces undersegmentation within the ESRG, but it is not "
         "sufficient on its own and is useful only in combination with a stopping criterion."))
    e5p, e5d, e6p, e6d = t("E5a", "precision"), t("E5a", "dsc"), t("E6", "precision"), t("E6", "dsc")
    p("Third, the stopping criterion that adapts to the intensity variability of the growing region, supported with a "
      "global drift guard, decreased boundary leakage in every comparison. Removed from the ESRG, its absence let the "
      f"region flood the brain on {leak.get('P_ESRG_C_noS', float('nan')):.0f}% of slices and lowered the median DSC from "
      f"{f3(e6d['enh']['median'])} to {f3(e6d['base']['median'])}. From the tumor seed alone, it kept the original "
      f"algorithm inside the tumor more strictly than two competing seeds did (median precision "
      f"{f3(e5p['enh']['median'])} against {f3(e5p['base']['median'])}), but it also stopped the region before the "
      f"whole tumor was covered, so the DSC was lower (median {f3(e5d['enh']['median'])} against "
      f"{f3(e5d['base']['median'])}). Objective 3 is "
      + ("therefore met." if V["pooled"]["objective3"] else
         "therefore met with respect to leakage, measured by precision, but not under the full decision rule, which "
         "also required a lower boundary distance and an unharmed DSC."))
    p(f"Taken together, the enhancements are complementary rather than independent: the local log-domain measure "
      f"lets the region reach the tumor boundary, and the stopping criterion keeps it from crossing it. The complete "
      f"ESRG with a planted seed reached a mean DSC of {f3(m('P_ESRG', 'pooled'))}, against "
      f"{f3(m('P_SRG', 'pooled'))} for the original algorithm supplied with tumor, healthy-tissue, and background "
      f"seeds, and {f3(m('A_ESRG', 'pooled'))} with automatic seeding. "
      + ("The ESRG therefore improves on the original algorithm. " if overall else
         "The study therefore does not demonstrate an overall improvement in segmentation accuracy over the original "
         "algorithm. ")
      + f"The ESRG leaks less (mean precision {f3(m('P_ESRG', 'pooled', met='precision'))} against "
        f"{f3(m('P_SRG', 'pooled', met='precision'))}) and needs only a single seed, or none, but it undersegments "
        f"more (mean recall {f3(m('P_ESRG', 'pooled', met='recall'))} against {f3(m('P_SRG', 'pooled', met='recall'))}). "
        f"Its practical advantage lies in the input it requires: given the same single tumor click, the original "
        f"algorithm absorbed the whole image (median DSC {f3(s1('dsc')['base']['median'])}), whereas the ESRG stopped at "
        f"the tumor (median DSC {f3(s1('dsc')['enh']['median'])}), a secondary comparison defined after the test run. "
        f"Performance depended strongly on the tumor type: meningioma was delineated well without any user input "
        f"(median DSC {f3(m('A_ESRG', 'meningioma', 'median'))}), whereas glioma, whose heterogeneous and infiltrative "
        "enhancement offers no sharp intensity boundary, remains the practical limit of the method.")
    p("These conclusions are subject to the following limitations. The system operates on single two-dimensional "
      "slices of one MRI sequence and uses the tumor core as the reference standard, excluding edema. Only slices "
      "with a tumor core of at least 150 mm² were eligible, so small tumors and the thin upper and lower sections of a "
      "tumor were not evaluated. The planted seeds were placed by a simulated careful operator with knowledge of the "
      "ground truth, so seeds placed by real users may give lower scores, particularly for the SRG, which needs three "
      "correctly placed seeds. The images were skull-stripped by the dataset providers, which the brain mask of the "
      "system relies on. Finally, the decision to adopt the running global mean was made by the researchers although "
      "it did not meet the pre-set development-set acceptance margin; it was documented before the test set was "
      "opened.")
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
    return {"title": "Chapter 5: Conclusions and Recommendations", "author": "Karylle Vinces J. Aliswag & Juana Mari A. Del Rosario", "blocks": B}


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
