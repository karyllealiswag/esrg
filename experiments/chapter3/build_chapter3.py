"""
build_chapter3.py — Chapter 3 (Methodology) of the ESRG thesis, BraTS 2023 version.

Purpose : Write the methodology that matches the final study design (experiments/PROTOCOL_BRATS.md,
          IMPROVEMENT_PROTOCOL_ESRG*.md): dataset and sampling, metrics and statistical treatment,
          experiments, the ESRG procedure, parameters, requirements, and the theories, methods and
          tools of each objective.
Function : builds a JSON block specification and renders it with experiments/chapter4/render_docx.js
          (US Letter, Times New Roman 12 pt, 1.5 spacing, APA tables, boxed numbered equations).
          Every equation is introduced in four parts (definition, purpose, meaning of the result,
          equation with its legend). Equation numbers are assigned automatically in the order the
          equations appear, so text references never go stale (the spec is built twice: the first
          pass collects the numbers). Numbers that come from the data (pool sizes, sample sizes,
          margins, tuned parameters) are read from the project files, never typed by hand.
          CLI: python -P experiments/chapter3/build_chapter3.py   -> Chapter3_Methodology.docx
"""
import csv
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from esrg import Config

OUT_DIR = os.path.join(ROOT, "outputs", "chapter3")
DOCX = os.path.join(ROOT, "Chapter3_Methodology.docx")
CFG = Config.load(os.path.join(ROOT, "configs", "esrg_brats_frozen_v3.json"))
Z = 1.959964
PER_TYPE = 300                                   # evaluated TEST sample per tumor type
MM_PER_PX = 240 / 512
NUM = {}                                         # equation key -> number, filled by the first pass


def pool_counts():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "brats_eval", "split.csv"))))
    out = {}
    for t in ("glioma", "meningioma"):
        r = [x for x in rows if x["tumor"] == t]
        out[t] = {"N": len(r), "dev": sum(x["set"] == "dev" for x in r), "test": sum(x["set"] == "test" for x in r)}
    return out


def cochran(N, e=0.05, p=0.5):
    n0 = Z * Z * p * (1 - p) / e ** 2
    return n0, n0 / (1 + (n0 - 1) / N)


def margin(n, N, p=0.5):
    return Z * math.sqrt(p * (1 - p) / n * (N - n) / (N - 1))


def build():
    P = pool_counts()
    Ng, Nm = P["glioma"]["N"], P["meningioma"]["N"]
    n0, ng = cochran(Ng)
    _, nm = cochran(Nm)
    V = {  # values substituted into the text as @v:name@
        "n0": f"{n0:.2f}", "ng": str(math.ceil(ng)), "nm": str(math.ceil(nm)), "Ng": f"{Ng:,}", "Nm": f"{Nm:,}",
        "nh": str(PER_TYPE), "ntot": str(2 * PER_TYPE),
        "mg": f"{100 * margin(PER_TYPE, Ng):.1f}", "mm": f"{100 * margin(PER_TYPE, Nm):.1f}",
        "gdev": str(P["glioma"]["dev"]), "mdev": str(P["meningioma"]["dev"]),
        "gtest": str(P["glioma"]["test"]), "mtest": str(P["meningioma"]["test"]),
        "mmpx": f"{MM_PER_PX:.3f}", "eta": f"{CFG.interior_frac:g}", "cmin": f"{CFG.min_clearance:g}",
        "amin": str(CFG.min_cand_area), "K": str(CFG.otsu_classes), "alpha": f"{CFG.seed_core_frac:g}",
        "wc": f"{CFG.rank_w_contrast:g}", "wr": f"{CFG.rank_w_radius:g}", "ws": f"{CFG.rank_w_shape:g}",
        "eps": f"{CFG.log_eps:g}", "sigmin": f"{CFG.sigma_floor_min:g}", "r": str(CFG.local_radius),
        "win": f"{2 * CFG.local_radius + 1} × {2 * CFG.local_radius + 1}", "kL": f"{CFG.k_local:g}",
        "kG": f"{CFG.k_global:g}", "sigcap": f"{CFG.sigma_cap:g}", "phi": f"{CFG.flood_frac:.3f}",
        "lam": f"{CFG.flood_shrink:g}", "tmax": str(CFG.flood_tries), "pmax": str(CFG.max_passes),
        "maxside": str(CFG.max_side), "omega": str(CFG.rank_contrast_ring),
    }
    B = []
    cnt = [0]

    def fmt(t):
        t = re.sub(r"@eq:(\w+)@", lambda m: f"Equation {NUM.get(m.group(1), '?')}", t)
        return re.sub(r"@v:(\w+)@", lambda m: V[m.group(1)], t)

    h1 = lambda t: B.append({"t": "h1", "text": t})
    h2 = lambda t: B.append({"t": "h2", "text": t})
    h3 = lambda t: B.append({"t": "h3", "text": t})
    h4 = lambda t: B.append({"t": "h4", "text": t})
    label = lambda t: B.append({"t": "label", "text": t})
    p = lambda t, **k: B.append({"t": "p", "text": fmt(t), **k})
    step = lambda t, lv=0: B.append({"t": "step", "text": fmt(t), "level": lv})
    boxed = lambda lab, items: B.append({"t": "box", "label": lab, "items": [{"text": fmt(t), "level": lv} for t, lv in items]})
    li = lambda n, t: B.append({"t": "li", "n": n, "text": fmt(t)})
    sub = lambda b, s: {"sub": [b, s]}
    sup = lambda b, s: {"sup": [b, s]}

    def eq(key, title, math_nodes):
        cnt[0] += 1
        NUM[key] = f"3.{cnt[0]}"
        B.append({"t": "eq", "num": NUM[key], "title": title, "math": math_nodes})

    def where(*pairs):
        """The legend, as one sentence directly under the equation it explains."""
        def verb(sym):
            return "are" if (" and " in sym or (", " in sym and "(" not in sym)) else "is"
        parts = [fmt(f"{s} {verb(s)} {d}") for s, d in pairs]
        body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
        B.append({"t": "p", "text": "where " + body + ".", "noindent": True})

    def table(num, title, header, rows, widths, note=None, align=None):
        B.append({"t": "table", "num": num, "title": title, "header": header, "rows": rows, "widths": widths,
                  "note": fmt(note) if note else None, **({"align": align} if align else {})})

    # ═══ Chapter heading ═════════════════════════════════════════════════════
    h1("Chapter Three")
    h1("METHODOLOGY")
    p("This chapter presents the research design, the dataset and its sampling, the evaluation metrics and "
      "statistical treatment, the proposed Enhanced Seeded Region Growing (ESRG) algorithm and its system "
      "architecture, the hardware and software requirements, and the theories, methods, and tools that "
      "implement each objective.")

    # ═══ 3.1 Research design ═════════════════════════════════════════════════
    h2("3.1 Research Design")
    p("This study employs a quantitative, experimental, and applied research design. Quantitative research "
      "tests objective theories by examining relationships among measurable variables, and experimental "
      "research determines whether a treatment influences an outcome by comparing a treatment group with a "
      "control group (Creswell & Creswell, 2023). The study is applied because it addresses the practical "
      "need for consistent, automated, and interpretable localization and delineation of brain tumors on "
      "two-dimensional (2D) contrast-enhanced T1-weighted (T1-CE) magnetic resonance imaging (MRI) slices.")
    p("The control group is the original Seeded Region Growing (SRG) algorithm of Adams and Bischof (1994), "
      "and the experimental group is the ESRG. The independent variables are the three enhancements: "
      "automatic seed selection (Objective 1), the local difference measure computed in the log domain "
      "(Objective 2), and the adaptive stopping criterion (Objective 3). The dependent variables are the "
      "segmentation-quality metrics of Section 3.1.2.")
    p("Both algorithms are run under identical conditions: the same slices, the same seeds, the same parameter "
      "values for every applied method, and the same evaluation procedure. The baseline SRG is implemented as "
      "published: each candidate pixel is compared with the mean intensity of the adjacent region, the "
      "immediate neighbors of a pixel are its 8-connected neighbors (Adams & Bischof, 1994, Section II), and "
      "every pixel is absorbed until the whole image has been allocated. Because the original algorithm stops "
      "a region only where a competing region reaches a pixel first, it is given competing seeds. The ESRG "
      "grows from the tumor seed alone, inside the brain, under the stopping rules of Section 3.4.3.")
    p("Seeding is treated as a controlled condition. Under automatic seeding, the tumor seed is selected by "
      "the system (Objective 1). Under planted seeding, a simulated careful operator places one seed of each "
      "type: a tumor seed at the deepest pixel of the ground-truth tumor core (placement P1), a healthy-tissue "
      "seed in the opposite hemisphere at least 25 mm from the tumor core, and a background seed in the air at "
      "least 8 pixels outside the brain, each a disk of radius 3 pixels. Two further placements, P2 and P3, "
      "move every seed by 4 to 8 mm within its own region, so that the dependence of a result on the exact "
      "click position can be measured. The ground-truth mask is used only to position the simulated seeds; "
      "the automatic seed never sees it.")
    p("The applied methods are switched on and off one at a time from the same seeds. When an enhancement is "
      "applied to the SRG baseline, it acts on the tumor region only and uses exactly the parameter values of "
      "the ESRG, so the SRG receives no separate tuning. The configurations that result are named and "
      "explained with the experiments in Section 3.1.2.")

    # ── 3.1.1 Dataset ────────────────────────────────────────────────────────
    h3("3.1.1 Dataset")
    p("The study uses the adult glioma (GLI) and meningioma (MEN) collections of the Brain Tumor Segmentation "
      "(BraTS) 2023 challenge (Baid et al., 2021; Bakas et al., 2017; LaBella et al., 2023; Menze et al., "
      "2015). Each case provides co-registered, skull-stripped multiparametric MRI volumes at an isotropic "
      "resolution of 1 mm with expert ground-truth labels. Only the post-contrast T1-weighted (T1-CE) volume is "
      "used, because contrast enhancement delineates the active tumor. The reference standard is the tumor "
      "core, the union of labels 1 (necrotic and non-enhancing tumor core) and 3 (enhancing tumor); the "
      "peritumoral edema (label 2) is excluded because it is not visible as tumor on T1-CE images.")
    p("For each patient, one axial slice was exported as a PNG image of at most @v:maxside@ × @v:maxside@ pixels "
      "together with its tumor-core mask, so that one pixel corresponds to 240/512 ≈ @v:mmpx@ mm. A slice was "
      "eligible if its tumor core covered at least 150 pixels on the native 240-pixel grid (about 150 mm²), and "
      "the exported slice was chosen at random among the eligible slices, so that the largest, easiest section "
      "is not selected systematically. Intensities are rescaled to 0–255 between the 0.5th and 99.5th "
      "percentiles, and the brain mask is the non-zero region that skull stripping leaves. These steps belong "
      "to the data, not to the algorithm. Every patient contributes exactly one slice, so the slices are "
      "independent observations and the unit of analysis is the patient.")

    h4("Sampling")
    p("The evaluated sample was drawn by stratified simple random sampling. The test set, the locked 70% of the "
      "pool (Table 3.1), is divided into two strata, glioma and meningioma, and an equal number of patients is "
      "drawn at random, without replacement, from each stratum with the fixed random seed 2026. Stratification "
      "guarantees that both tumor types are represented, and equal allocation estimates both with the same "
      "precision. The drawn list was recorded before the evaluation started.")
    p("**Cochran’s sample size formula** (Cochran, 1977) gives the number of patients needed to estimate a "
      "proportion, here the seed hit rate of Objective 1, within a chosen margin of error at 95% confidence. "
      "Its purpose is to justify the sample size statistically. A smaller margin of error requires a larger "
      "sample, and p = 0.5 is the most conservative assumption because it gives the largest sample.")
    eq("cochran", "Sample Size Formula for an Unlimited Population",
       [sub("n", "0"), " = ", {"f": [[sup("z", "2"), "p", {"rb": "1 − p"}], sup("e", "2")]}])
    where(("*n*_{0}", "the sample size for an unlimited population"),
          ("*z*", "the standard normal value for 95% confidence, 1.96"),
          ("*p*", "the expected proportion, 0.5"), ("*e*", "the margin of error, 0.05"))
    p("**The finite population correction** reduces the required sample when the pool is limited. Its purpose "
      "is to avoid sampling more patients than the pool warrants. The smaller the pool, the larger the "
      "reduction.")
    eq("fpc", "Finite Population Correction Formula",
       ["n = ", {"f": [sub("n", "0"), ["1 + ", {"f": [[sub("n", "0"), " − 1"], "N"]}]]}])
    where(("*n*", "the minimum sample size of the tumor type"),
          ("*N*", "the number of patients of that tumor type in the pool"))
    p("The pool, not the test set, is the population to which the results are generalized; because the test set "
      "is a random 70% of the pool, a random sample of the test set is a random sample of the pool. With "
      "e = 0.05, the first equation gives n_{0} = @v:n0@, and the second gives a minimum of @v:ng@ glioma "
      "patients (N = @v:Ng@) and @v:nm@ meningioma patients (N = @v:Nm@). The larger requirement, @v:ng@, was "
      "applied to both strata and increased by about 9% as an allowance for cases that cannot be evaluated "
      "(Cochran, 1977), giving n_{h} = @v:nh@ patients per tumor type and @v:ntot@ in total. The resulting "
      "margins of error at 95% confidence are ±@v:mg@ percentage points for glioma and ±@v:mm@ for meningioma.")

    h4("Dataset Presented")
    p("Table 3.1 presents the dataset. The pool was split once at the patient level, separately for each tumor "
      "type and with the fixed random seed 2026, into a development set (30%) and a test set (70%). All "
      "parameter tuning used the development set only. The test set was locked until the final configuration "
      "had been frozen, and the sample drawn from it was evaluated once; the full record of tuning decisions "
      "is kept in the study protocol.")
    table("3.1", "Number of Slices per Tumor Type in the Pool, the Development and Test Sets, and the Evaluated Sample",
          ["Tumor type", "Pool (N)", "Development set", "Test set", "Minimum n", "Evaluated sample (n_{h})"],
          [["Glioma", V["Ng"], V["gdev"], V["gtest"], V["ng"], V["nh"]],
           ["Meningioma", V["Nm"], V["mdev"], V["mtest"], V["nm"], V["nh"]],
           ["Total", f"{Ng + Nm:,}", str(P["glioma"]["dev"] + P["meningioma"]["dev"]),
            f'{P["glioma"]["test"] + P["meningioma"]["test"]:,}', "—", V["ntot"]]],
          [1.3, 1.0, 1.3, 1.0, 1.1, 1.5],
          note=("Each count is one slice and one patient. The development and test sets are the 30%/70% patient-level "
                "split of the pool; the evaluated sample is drawn from the test set. Minimum n is the result of the "
                "finite population correction for the tumor type."))

    # ── 3.1.2 Metrics ────────────────────────────────────────────────────────
    h3("3.1.2 Evaluation Metrics")
    p("Each predicted tumor mask M is compared with the expert ground-truth tumor-core mask G at the working "
      "resolution of the slice. Each objective is judged by the metric that responds most directly to its "
      "problem (Maier-Hein et al., 2024; Taha & Hanbury, 2015): the seed hit rate for Objective 1, recall for "
      "undersegmentation (Objective 2), and precision and HD95 for boundary leakage (Objective 3). The Dice "
      "similarity coefficient is the overall measure, and the processing time per slice measures efficiency. The overlap metrics are computed from three pixel counts: "
      "the true positives (TP), tumor in both M and G; the false positives (FP), tumor only in M; and the false "
      "negatives (FN), tumor only in G.")

    p("**Seed Hit Rate (SHR)** is the percentage of slices on which the entire automatic seed lies inside the "
      "tumor core. It measures how reliably automatic seed selection starts growth inside the tumor. A higher "
      "value is better; a slice on which no seed is found counts as a miss. The target of Objective 1 is a lower "
      "confidence bound of at least 70%.")
    eq("shr", "Seed Hit Rate Formula", ["SHR = ", {"f": [sub("n", "hit"), "n"]}, " × 100%"])
    where(("*n*_{hit}", "the number of slices whose seed core lies entirely inside the tumor core"),
          ("*n*", "the number of slices evaluated"))

    p("**Recall** is the share of the true tumor that the segmentation recovers. It measures undersegmentation "
      "(Objective 2). A higher value is better: a value of 1 means no tumor was missed, and a low value means "
      "part of the tumor was left out.")
    eq("recall", "Recall Formula", ["Recall = ", {"f": ["TP", "TP + FN"]}])
    where(("TP", "the number of tumor pixels found by the segmentation"),
          ("FN", "the number of tumor pixels the segmentation missed"))

    p("**Precision** is the share of the segmented region that is truly tumor. It measures boundary leakage "
      "(Objective 3). A higher value is better: a value of 1 means no non-tumor tissue was included, and a low "
      "value means the region leaked into the surrounding tissue.")
    eq("precision", "Precision Formula", ["Precision = ", {"f": ["TP", "TP + FP"]}])
    where(("TP", "the number of tumor pixels found by the segmentation"),
          ("FP", "the number of non-tumor pixels included in the segmentation"))

    p("**Dice Similarity Coefficient (DSC)** measures the overall overlap between the segmentation and the "
      "ground truth. It rewards a method only when both undersegmentation and leakage are controlled, so it "
      "serves as the overall measure and as a no-harm check, which confirms that an improvement in one error "
      "was not obtained by a larger error of the other type. Values range from 0 (no overlap) to 1 (perfect "
      "overlap); a higher value is better, and 0.70 or above is conventionally regarded as good agreement "
      "(Zijdenbos et al., 1994).")
    eq("dsc", "Dice Similarity Coefficient Formula", ["DSC = ", {"f": ["2TP", "2TP + FP + FN"]}])
    where(("TP, FP, and FN", "the pixel counts defined above"))

    p("**HD95**, the 95th-percentile Hausdorff distance, measures how far the boundary of the segmentation lies "
      "from the true tumor boundary. A lower value is better, and a leak produces a high value because it "
      "creates boundary pixels far from the true edge. It is computed separately for every slice: the distance "
      "from each boundary pixel of one mask to the nearest boundary pixel of the other mask is measured in both "
      "directions (Huttenlocher et al., 1993), the distances are combined into one list, and HD95 is the 95th "
      "percentile of that list. It is therefore one percentile of one list per slice, not a mean. The largest "
      "distance (the classical Hausdorff distance) can be set by a single stray pixel, whereas the 95th "
      "percentile ignores the most extreme 5% of the distances. HD95 is given in pixels (1 pixel = @v:mmpx@ mm); "
      "an empty segmentation receives the image diagonal, the worst possible distance (Maier-Hein et al., 2024).")
    eq("hd95", "HD95 Formula",
       ["HD95 = ", sub("P", "95"), {"rb": "D"}, ",   D = {d(p, ∂G) : p ∈ ∂M} ∪ {d(q, ∂M) : q ∈ ∂G}"])
    where(("∂M and ∂G", "the boundary pixels of the segmentation and of the ground truth"),
          ("d(p, ∂G)", "the Euclidean distance from pixel p to the nearest pixel of ∂G"),
          ("*P*_{95}", "the 95th percentile of the list D"))

    p("**Processing Time** is the time the system needs to segment one slice, from the start of the run to the "
      "finished mask, without the scoring against the ground truth. It measures computational efficiency. A lower "
      "value is better. Because the batch evaluation runs several slices in parallel on one laptop, the times indicate "
      "the relative speed of the configurations, not their absolute speed.")
    eq("time", "Processing Time Formula", ["t = ", sub("t", "end"), " − ", sub("t", "start")])
    where(("*t*", "the processing time of one slice, in seconds"),
          ("*t*_{start} and *t*_{end}", "the clock readings at the start and at the end of the segmentation"))

    h4("Statistical Treatment")
    p("Results are reported separately for glioma and meningioma, because the two tumor types differ markedly "
      "in how they enhance. Because per-slice DSC values are bounded and often skewed, each configuration is "
      "described by its median and interquartile range together with its mean and standard deviation. Every "
      "configuration was run on the same slices, so configurations are compared with paired tests. Differences "
      "are oriented so that a positive value favors the enhanced configuration; for HD95 and processing time, "
      "where lower is better, the sign is reversed. All tests are two-sided at α = .05.")
    p("**The Shapiro–Wilk test** (Shapiro & Wilk, 1965) checks whether the paired differences are normally "
      "distributed. A p value below .05 means they are not, which justifies the rank-based tests below instead "
      "of a paired t-test.")
    p("**The Wilcoxon signed-rank test** (Wilcoxon, 1945) compares two paired configurations from the ranks of "
      "their per-slice differences. A p value below .05 indicates a systematic difference, and the sign of the "
      "median difference shows which configuration is better. Because a negligible difference can reach "
      "significance with many slices, the effect size r is also reported. A value of |r| near .1, .3, and .5 "
      "indicates a small, medium, and large effect (Fritz et al., 2012); a positive r favors the enhanced "
      "configuration.")
    eq("effect", "Wilcoxon Effect Size Formula", ["r = ", {"f": ["Z", {"sqrt": "N"}]}])
    where(("*Z*", "the standardized Wilcoxon statistic"), ("*N*", "the number of paired slices"))
    p("**The Friedman test** (Friedman, 1937) compares more than two related configurations on the same slices, "
      "namely the four configurations of E1 and the automatic and planted placements of E2. Each slice ranks "
      "the configurations, and a p value below .05 means at least one configuration differs from the others; "
      "pairwise Wilcoxon tests then show which.")
    eq("friedman", "Friedman Test Statistic",
       [{"subsup": ["χ", "F", "2"]}, " = ", {"f": ["12", "Nk(k + 1)"]},
        {"sum": {"lo": "j=1", "hi": "k", "body": [{"subsup": ["R", "j", "2"]}]}}, " − 3N(k + 1)"])
    where(("*N*", "the number of slices"), ("*k*", "the number of configurations"),
          ("*R*_{j}", "the sum of the ranks of configuration j over all slices"))
    p("**Kendall’s W** (Kendall & Babington Smith, 1939) expresses the Friedman result as an effect size from 0 "
      "to 1. A value near 0 means the ranking changes from slice to slice, so there is no consistent "
      "difference; a value near 1 means the configurations are ordered the same way on almost every slice.")
    eq("kendall", "Kendall’s W Formula", ["W = ", {"f": [{"subsup": ["χ", "F", "2"]}, "N(k − 1)"]}])
    where(("χ_{F}^{2}", "the Friedman statistic"), ("*N* and *k*", "the number of slices and of configurations"))
    p("**The Wilson score interval** (Wilson, 1927) gives the 95% confidence interval of the seed hit rate. It "
      "remains accurate for proportions near 0% and 100%, and it narrows as more slices are evaluated. "
      "Objective 1 requires its lower bound to be at least 70%.")
    eq("wilson", "Wilson Score Interval Formula",
       ["CI = ", {"f": [["p̂ + ", {"f": [sup("z", "2"), "2n"]}, " ± z", {"sqrt": [{"f": ["p̂(1 − p̂)", "n"]}, " + ", {"f": [sup("z", "2"), "4n²"]}]}],
                         ["1 + ", {"f": [sup("z", "2"), "n"]}]]}])
    where(("p̂", "the observed proportion of hits"), ("*n*", "the number of slices"),
          ("*z*", "1.96 for 95% confidence"))
    p("**The Holm adjustment** (Holm, 1979) corrects the p values when several tests are made in one experiment, "
      "for example for recall and DSC, so that the chance of a false positive across all of them stays at 5%. "
      "A result counts as significant only if its adjusted p value is below .05.")
    eq("holm", "Holm Adjusted p Value Formula",
       [{"sup": [sub("p", "(i)"), "adj"]}, " = max", sub("", "j ≤ i"), " min{1, (m − j + 1) ", sub("p", "(j)"), "}"])
    where(("*m*", "the number of tests in the experiment"),
          ("*p*_{(j)}", "the j-th smallest p value of the experiment"))

    p("The decision rules were fixed before the test set was evaluated and are applied separately to each "
      "tumor type. An effect counts only if its adjusted p value is below .05 and |r| ≥ 0.1. The overall "
      "improvement is established if the DSC of both A_ESRG and P_ESRG is significantly higher than that of "
      "P_SRG. Objective 1 is met if the lower bound of the Wilson interval of the seed hit rate is at least 70%, "
      "the automatic result is identical across repeated runs while the planted results differ across "
      "placements, and the automatic DSC is not significantly lower than that of any planted placement. "
      "Objective 2 is met if recall is significantly higher and DSC not significantly lower in both E3 and E4. "
      "Objective 3 is met if precision is significantly higher, HD95 significantly lower, and DSC not "
      "significantly lower in both E5a and E6.")

    h4("Summary of Experiments")
    p("Each objective is tested from two directions where possible: by adding the method to the baseline SRG, "
      "which asks whether the method is sufficient, and by removing it from the ESRG, which asks whether the "
      "method is necessary. The configurations in Table 3.2 are named as follows.")
    B.append({"t": "legend", "items": [
        ["A_", "the tumor seed is selected automatically"],
        ["P_", "the tumor seed is planted by the simulated operator at placement P1 (P2 and P3 are marked _2 and _3)"],
        ["SRG, ESRG", "the algorithm that is run"],
        ["_L", "the local log-domain difference measure (LLDM) is added to the SRG"],
        ["_S", "the adaptive stopping criterion (ASC) is added to the SRG"],
        ["_noL, _noS", "LLDM or ASC is removed from the ESRG"],
        ["_solo", "the SRG receives the tumor seed only, without competing seeds"],
        ["_C", "the ESRG is run with competing seeds, as the SRG is"]]})
    table("3.2", "Summary of Experiments",
          ["Experiment", "Purpose", "Configurations", "Metrics", "Statistical tests"],
          [["E1 Overall and seeding", "Complete algorithms at different operator input (none, one seed, three seeds), effect of automatic seeding, seed hit rate (Objective 1)", "A_ESRG, P_ESRG, P_SRG_solo, P_SRG", "DSC (primary), recall, precision, HD95, time; SHR", "Friedman, Kendall’s W, Wilcoxon with Holm; Wilson interval"],
           ["E2a, E2b Seeding consistency (Objective 1)", "Dependence of the result on seed placement; automatic vs. planted", "A_ESRG or A_SRG; P_SRG, P_SRG_2, P_SRG_3", "DSC", "Friedman, Kendall’s W, Wilcoxon with Holm"],
           ["E3 LLDM added (Objective 2)", "Sufficiency of LLDM in the baseline", "P_SRG → P_SRG_L", "Recall (primary); DSC (no-harm)", "Wilcoxon with Holm"],
           ["E4 LLDM removed (Objective 2)", "Necessity of LLDM in the ESRG", "P_ESRG_noL → P_ESRG", "Recall (primary); DSC (no-harm)", "Wilcoxon with Holm"],
           ["E5a ASC replaces competitors (Objective 3)", "Whether ASC with a single seed matches SRG with three seeds", "P_SRG → P_SRG_solo_S", "Precision, HD95 (primary); DSC (no-harm)", "Wilcoxon with Holm"],
           ["E5b ASC as a stopping rule (Objective 3)", "Effect of ASC when SRG has no competitors", "P_SRG_solo → P_SRG_solo_S", "Precision, HD95", "Wilcoxon with Holm"],
           ["E6 ASC removed (Objective 3)", "Necessity of ASC in the ESRG", "P_ESRG_C_noS → P_ESRG_C", "Precision, HD95 (primary); DSC (no-harm)", "Wilcoxon with Holm"],
           {"group": "Secondary comparison (added after the test run; descriptive)"},
           ["S2 Same automatic seed", "Both algorithms started from the same automatic tumor seed", "A_SRG → A_ESRG", "DSC, precision, recall", "Wilcoxon with Holm"]],
          [1.6, 2.0, 1.6, 1.5, 1.5],
          note=("→ denotes baseline → enhanced. The ESRG configurations without a suffix run with LLDM and ASC on. "
                "The baseline of E5b absorbs the whole image by design, so it illustrates the stopping rule rather "
                "than testing it against a competent baseline. The upper bound σ_{cap} is omitted in the ASC-only "
                "configurations because it is calibrated in log units. Determinism is verified by running A_ESRG and "
                "A_SRG twice more and comparing the masks. P_SRG_solo, which is the baseline of E5b, joined the overall "
                "comparison E1 after the test run, to compare the algorithms at equal input, and S2 was defined after "
                "the test run; both use results already collected, and the decision rules still use P_SRG as the baseline."),
          align=["l", "l", "l", "l", "l"])

    # ═══ 3.2 Algorithm and architecture ══════════════════════════════════════
    h2("3.2 Proposed Algorithm and System Architecture")
    p("This section presents the ESRG as the modification of the SRG by the three enhancements, and describes "
      "how the procedure is organized into an Input–Process–Output system.")
    h3("3.2.1 Enhanced Seeded Region Growing Algorithm")
    p("The ESRG procedure is given below in general form. It keeps the growing loop of the SRG, in which the "
      "pixels adjacent to the region are held in a list sorted by their difference measure and the most similar "
      "pixel is examined first (Adams & Bischof, 1994), and it replaces or extends that loop with the steps "
      "enclosed in a box, each of which implements the objective indicated. The specific algorithm of each step "
      "is given under the corresponding objective in Section 3.4, and the parameter values are listed in "
      "Table 3.3.")
    step("START")
    step("Input: an MRI slice and, optionally, seed clicks. Output: the tumor region A.")
    boxed("OBJECTIVE 1", [("1. Select the seed S automatically from the slice; if the user supplies seeds, use them as S.", 0)])
    boxed("OBJECTIVE 2", [("2. Express the intensity of every pixel in the log domain.", 0)])
    step("3. Set the region A = S. Repeat the following growing pass until no pixel is added to A:")
    boxed("OBJECTIVE 3", [("a. Set the acceptance bounds from the intensity variability of A: the local bound T_{L} and the global bound T_{G}.", 1)])
    boxed("OBJECTIVE 2", [("b. For each pixel x adjacent to A, compute its local log-domain difference measure δ(x) and insert x into the sequentially sorted list (SSL) in ascending order of δ.", 1)])
    step("c. While the SSL is not empty, remove the pixel y with the smallest δ:", 1)
    boxed("OBJECTIVE 3", [("i. If δ(y) > T_{L}, end the pass.", 2),
                          ("ii. If the log intensity of y differs from the running mean m_{A} of A by more than T_{G}, reject y and take the next pixel.", 2)])
    step("iii. Otherwise, add y to A, update the statistics of A, and insert the neighbors of y into the SSL.", 2)
    boxed("OBJECTIVE 3", [("4. If A covers more than a fraction φ of the brain, tighten the bounds and regrow from S.", 0)])
    step("5. Output A as the tumor region.")
    step("END")
    p("When manual seeding is selected in the interface, Step 1 is replaced by the user’s clicks, so that the ESRG "
      "and the SRG start from identical seeds. The symbols of the procedure are defined with the equations of "
      "Section 3.4. The original SRG corresponds to the same loop without the boxed steps, with the difference "
      "measured against the mean of the whole region and with every pixel eventually absorbed.")
    table("3.3", "Parameters of the Enhanced Seeded Region Growing Algorithm",
          ["Symbol", "Description", "Value"],
          [{"group": "Automatic seed selection (Objective 1)"},
           ["K", "Number of multi-level Otsu classes", V["K"]],
           ["η", "Interior depth fraction of the maximum brain depth", V["eta"]],
           ["A_{min}", "Minimum candidate component area, in pixels", V["amin"]],
           ["c", "Minimum clearance of a candidate from the brain outline, in pixels", V["cmin"]],
           ["ω", "Width of the surrounding ring used for local contrast, in pixels", str(CFG.rank_contrast_ring)],
           ["w_{c}, w_{r}, w_{s}", "Ranking weights of contrast, inscribed radius, and shape", f"{V['wc']}, {V['wr']}, {V['ws']}"],
           ["α", "Seed core fraction of the maximum distance-transform value", V["alpha"]],
           ["—", "Radius of the disk around each seed click, in pixels", str(CFG.manual_seed_radius)],
           {"group": "Log-domain transformation and local measure (Objective 2)"},
           ["ε", "Log-transform offset", V["eps"]],
           ["σ_{min}", "Lower bound of the noise floor, in log units", V["sigmin"]],
           ["r", "Local window radius, giving a " + V["win"] + " window", V["r"]],
           {"group": "Adaptive stopping criterion (Objective 3)"},
           ["k_{L}", "Local confidence multiplier", V["kL"]],
           ["k_{G}", "Global drift-guard multiplier", V["kG"]],
           ["σ_{cap}", "Upper bound of the region standard deviation, in log units", V["sigcap"]],
           ["φ", "Flood-guard fraction of the brain area", V["phi"]],
           ["λ, t_{max}", "Flood-guard shrink factor and maximum number of retries", f"{V['lam']}, {V['tmax']}"],
           ["P_{max}", "Maximum number of growing passes", V["pmax"]],
           ["—", "Global reference of the drift guard", "Running region mean"]],
          [1.2, 4.0, 1.4],
          note=("All values were tuned on the development set and frozen before the test set was evaluated. The "
                "flood-guard fraction φ is the 95th percentile of the tumor-core-to-brain area ratio of the "
                "development ground truth. When an applied method is switched on in the baseline SRG, the SRG uses "
                "the same values. Research options implemented in the system but not part of the evaluated ESRG, "
                "namely Frangi vesselness ranking, anatomical exclusion of the orbits and skull base, a per-image "
                "adaptive k_{L}, and a healthy-tissue rule, are disabled."),
          align=["l", "l", "c"])
    p("Table 3.3 groups the parameters by the objective they serve. The values are fixed for every slice and "
      "tumor type, so the method requires no per-image adjustment; k_{G} must exceed k_{L}, and the system "
      "rejects any configuration that violates this rule.")

    h3("3.2.2 Proposed System Architecture")
    B.append({"t": "figure", "num": "3.1", "title": "Input–Process–Output Structure of the Proposed System",
              "img": "figure_3_1.png", "w": 2048, "h": 1070,
              "note": "The ground-truth mask enters only the Evaluation stage; it is never used during segmentation."})
    p("Figure 3.1 presents the architecture of the system in the Input–Process–Output (IPO) model. The Input is a "
      "skull-stripped, axial, contrast-enhanced T1-weighted 2D MRI slice stored as a PNG file.")
    p("The Process has two parts. In Pre-processing, the Log-Domain Transformation converts the pixel intensities "
      "to the log domain and estimates the noise floor, the smallest region variability that the image noise "
      "allows. Core Processing then contains the three enhancements: Automatic Seed Selection, which finds the "
      "tumor seed through multi-level Otsu thresholding, ranking of the candidates, and a seed core "
      "(Objective 1); Region Growing, which compares each pixel with its newly absorbed neighbors in the log "
      "domain (Objective 2); and the Adaptive Stopping Criterion, which bounds the growth with the region mean, "
      "the σ cap, and the flood guard (Objective 3).")
    p("The Output is the enhanced segmentation of the tumor mass, a binary mask that is displayed as an overlay "
      "on the slice. Before the mask is returned, its holes are filled, a one-pixel opening removes thin "
      "protrusions, and only the part connected to the seed is kept; these clean-up steps are not part of the "
      "enhancements. When no candidate survives seed selection, the system reports that no tumor candidate was "
      "found instead of placing an arbitrary seed.")
    p("The Evaluation compares the output with the ground-truth tumor-core mask using the Dice similarity "
      "coefficient, precision, recall, HD95, and the seed hit rate defined in Section 3.1.2.")

    # ═══ 3.3 Requirements ═════════════════════════════════════════════════════
    h2("3.3 System Requirements")
    p("This section specifies the hardware and software used to develop, run, and evaluate the system, and the "
      "steps required to install it.")
    h3("3.3.1 Hardware Requirements")
    p("The system was developed and evaluated on an Acer Nitro 5 laptop. Table 3.4 lists its specifications "
      "together with the minimum configuration recommended for running the system.")
    table("3.4", "Hardware Requirements",
          ["Component", "Specification used", "Recommended minimum", "Function"],
          [["Processor (CPU)", "Intel Core i5-11400H, 6 cores, 12 threads, 2.70 GHz base", "64-bit dual-core x86-64 CPU", "Executes all pipeline stages; the batch evaluation runs one slice per worker process"],
           ["Memory (RAM)", "8 GB DDR4-3200", "4 GB", "Holds the slice arrays, the SSL, and the evaluation records"],
           ["Storage", "512 GB NVMe SSD", "1 GB free space", "Stores the exported BraTS slices (about 75 MB), the code, and the outputs"],
           ["Graphics (GPU)", "NVIDIA GeForce RTX 3050 Laptop GPU", "Not required", "Not used; all computation runs on the CPU"],
           ["Display", "15.6-in., 1920 × 1080", "1280 × 720", "Shows the interface, whose minimum window size is 1080 × 680 pixels"],
           ["Input devices", "Keyboard and touchpad or mouse", "Keyboard and mouse", "Parameter entry, file selection, and manual seed clicks"],
           ["Operating system", "Windows 11 Home, 64-bit", "64-bit Windows, macOS, or Linux", "Runs Python and its scientific libraries"]],
          [1.2, 1.9, 1.5, 2.2],
          note="The model is the Acer Nitro 5 AN515-57. The batch evaluation used 11 worker processes.",
          align=["l", "l", "l", "l"])
    p("Table 3.4 shows that the system has modest requirements. The ESRG is a sequential, CPU-bound algorithm "
      "that processes one slice of at most 512 × 512 pixels at a time, so a GPU is unnecessary and 4 GB of "
      "memory suffices for interactive use. The multi-core processor is exploited only by the batch "
      "evaluation, which distributes slices across worker processes.")
    h3("3.3.2 Software Requirements")
    p("The system is written in Python and relies on open-source scientific libraries. Table 3.5 lists the "
      "software, the versions used in this study, and the minimum versions required.")
    table("3.5", "Software Requirements",
          ["Software", "Version used (minimum)", "Function"],
          [["Python", "3.14.4 (3.10)", "Implementation language of the system (Python Software Foundation, n.d.)"],
           ["NumPy", "2.4.5 (1.26)", "Array storage of slices and masks, percentiles, logarithms, and medians (Harris et al., 2020)"],
           ["SciPy", "1.17.1 (1.11)", "Connected-component labeling, hole filling, Euclidean distance transform, median filter, and statistical distributions (Virtanen et al., 2020)"],
           ["scikit-image", "0.26.0 (0.22)", "Otsu and multi-level Otsu thresholds, disk structuring elements, morphological opening, and region properties (van der Walt et al., 2014)"],
           ["Pillow", "12.2.0 (10.0)", "Image and mask loading, grayscale conversion, and resizing (Clark, 2015)"],
           ["pandas", "3.0.6 (2.1)", "Tabulation of per-slice results and statistical summaries (McKinney, 2010)"],
           ["Matplotlib", "3.11.2 (3.7)", "Figures of the evaluation (Hunter, 2007)"],
           ["SimpleITK", "2.5.5 (2.3)", "Reading the BraTS volumes when exporting the 2D slices (Lowekamp et al., 2013)"],
           ["Tkinter", "Bundled with Python (Tk 8.6)", "Desktop graphical user interface for loading slices, placing seeds, and inspecting each stage"],
           ["heapq", "Python standard library", "Binary-heap implementation of the sequentially sorted list"]],
          [1.3, 1.6, 3.8],
          note="Versions in parentheses are the minimum versions declared in the project's requirements file. SimpleITK is needed only to export slices from the BraTS volumes, not for segmentation.",
          align=["l", "l", "l"])
    p("Table 3.5 shows that segmentation itself depends only on NumPy, SciPy, scikit-image, and Pillow, while "
      "pandas, Matplotlib, and SimpleITK support the data preparation and evaluation. The system is installed "
      "and run as follows:")
    for i, t in enumerate([
            "Install Python 3.10 or later from python.org, including the Tcl/Tk component, and add Python to the system PATH.",
            "Copy the project folder to the computer and open a terminal in that folder.",
            "Optionally, create and activate a virtual environment: python -m venv .venv, then .venv\\Scripts\\activate on Windows or source .venv/bin/activate on macOS and Linux.",
            "Install the dependencies: pip install -r requirements.txt.",
            "Verify the installation by running the test suite: python tests/test_pipeline.py.",
            "Launch the interface with python -m gui.app, open an MRI slice, choose the method (ESRG or SRG) and the seeding mode (automatic or manual), and press RUN.",
            "To reproduce the study, run in order: python -P experiments/sampling.py --brats --draw (patient-level split), python -P experiments/tune.py --stage all (tuning on the development set), and python -P experiments/evaluate.py --set test --sample-per-type 300 --approved --analyze (the single test run and its analysis)."], 1):
        li(i, t)

    # ═══ 3.4 Theories, methods, tools ════════════════════════════════════════
    h2("3.4 Theories, Methods, and Tools")
    p("This section presents, for each objective, the theories that justify the enhancement, the methods and "
      "algorithms that implement it, and the tools used. The original SRG accepts an intensity image and a set "
      "of seeds and grows the seeded regions until all pixels are assimilated (Mehnert & Jackway, 1997), "
      "comparing each candidate pixel with the mean intensity of the adjacent region (Adams & Bischof, 1994); "
      "the enhancements change how the seed is obtained, what the comparison measures, and when a pixel is "
      "absorbed or rejected.")

    # ── 3.4.1 ────────────────────────────────────────────────────────────────
    h3("3.4.1 To provide an automatic seed selection option through Multi-Level Otsu Thresholding with connected-component filtering")
    h4("Theories")
    label("Optimal Histogram Thresholding")
    p("Otsu’s method separates an image into classes by intensity without operator input (Otsu, 1979). It "
      "chooses the thresholds that maximize the between-class variance of the gray-level histogram, which is "
      "equivalent to minimizing the variance inside each class. The multi-level extension divides the "
      "histogram into K classes with K − 1 thresholds, found efficiently by the recursive search of Liao et al. "
      "(2001). A larger between-class variance means the classes are better separated. On T1-CE images the "
      "enhancing tumor is among the brightest tissues, so it falls in the highest class.")
    eq("otsu", "Between-Class Variance Formula",
       [sup("σ", "2"), {"sub": ["", "B"]}, {"rb": [sub("t", "1"), ", …, ", sub("t", "K−1")]}, " = ",
        {"sum": {"lo": "k=1", "hi": "K", "body": [sub("ω", "k"), {"rb": [sub("μ", "k"), " − ", sub("μ", "T")]}, sup("", "2")]}}])
    where(("*t*_{1}, …, *t*_{K−1}", "the thresholds, with *t*_{K−1} separating the highest-intensity class"),
          ("*ω*_{k} and *μ*_{k}", "the pixel fraction and mean intensity of class k"),
          ("*μ*_{T}", "the mean intensity of all pixels"))
    label("Connected Components and Object Interiority")
    p("A binary mask can be divided into connected components, groups of foreground pixels linked through their "
      "neighbors, so that each candidate structure can be measured and accepted or rejected on its own "
      "(Gonzalez & Woods, 2018). The Euclidean distance transform gives each pixel its distance to the nearest "
      "pixel outside the component; its maximum lies at the center of the largest inscribed disk, a point that "
      "always lies inside the component, even for a ring-enhancing tumor, and avoids boundary pixels affected "
      "by the partial volume effect (Tohka, 2014).")
    label("Local Contrast of Enhancing Lesions")
    p("An enhancing lesion is brighter than its surroundings, which is the principle of the white top-hat "
      "transform (Soille, 2003). Automatic seeding from the brightest regions of brain MRI has been used for "
      "tumor segmentation on BraTS data (Biratu et al., 2021). Candidates are therefore ranked by their "
      "contrast against a surrounding ring, not by absolute brightness alone.")
    h4("Methods")
    label("Automated Seed Selection")
    p("Multi-level Otsu thresholding with K = @v:K@ is applied to the interior of the brain, the pixels whose "
      "depth from the brain outline exceeds both the minimum clearance c and a fraction η = @v:eta@ of the "
      "maximum depth. The candidate mask keeps the interior pixels of the highest class, and connected-component "
      "filtering separates the tumor from other bright structures. Components smaller than A_{min} = @v:amin@ "
      "pixels are discarded as noise. The remaining components are tested for interiority with the clearance of "
      "the following equation, which measures how far the component lies inside the brain beyond its own radius. "
      "A larger clearance means a more interior component, and a component is discarded if its clearance is "
      "below c. The tuned value is c = @v:cmin@, because on skull-stripped data no scalp tissue touches the brain "
      "outline, while meningiomas, which arise from the meninges, commonly do.")
    eq("clearance", "Clearance Formula",
       [sub("κ", "i"), " = depth", {"rb": sup(sub("x", "i"), "*")}, " − ", sub("ρ", "i")])
    where(("*x*_{i}^{*}", "the pixel of component i farthest from its boundary"),
          ("depth(*x*_{i}^{*})", "the distance of that pixel from the brain outline"),
          ("*ρ*_{i}", "the inscribed radius of component i, its largest distance-transform value"),
          ("*κ*_{i}", "the clearance of component i"))
    p("The surviving components are ranked by their local contrast Δ_{i}, the mean intensity of the component "
      "minus that of a surrounding ring of @v:omega@ pixels; their inscribed radius ρ_{i}; and a shape score "
      "s_{i} between 0 and 1 that averages compactness and solidity (Gonzalez & Woods, 2018). The ranking score "
      "is the weighted sum below, after contrast and radius are rescaled to [0, 1]; a higher score marks a "
      "better candidate, and the highest-ranked component is selected. The weights make interiority the "
      "dominant criterion, contrast secondary, and shape a tie-breaker. The seed core S is formed from the "
      "medial pixels of the selected component, those whose distance-transform value is at least α = @v:alpha@ "
      "of its maximum.")
    eq("ranking", "Candidate Ranking Score Formula",
       [sub("Q", "i"), " = ", sub("w", "c"), sup(sub("Δ", "i"), "′"), " + ", sub("w", "r"), sup(sub("ρ", "i"), "′"), " + ",
        sub("w", "s"), sub("s", "i")])
    where(("*Q*_{i}", "the ranking score of component i"),
          ("*Δ*_{i}′ and *ρ*_{i}′", "the rescaled contrast and inscribed radius"),
          ("*s*_{i}", "the shape score"),
          ("*w*_{c}, *w*_{r}, and *w*_{s}", "the weights @v:wc@, @v:wr@, and @v:ws@"))
    p("Seed selection remains optional: in manual mode each click is dilated into a disk of radius 3 pixels and "
      "used as it is, and the SRG additionally accepts clicks for its competing healthy-tissue and background "
      "regions. Under automatic seeding, the SRG uses the same automatic tumor seed with planted competing "
      "seeds. The seed-selection parameters were tuned on the development set by the seed hit rate, which does "
      "not depend on the growth rule.")
    label("Algorithm of Automated Seed Selection")
    step("START")
    step("Input: the normalized brain-masked slice, the brain mask and its depth map, and the parameters of Table 3.3. Output: the seed core S, or “no tumor candidate.”")
    step("1. Form the interior region of the brain; if it is empty or has constant intensity, return “no tumor candidate.”")
    step("2. Compute the K − 1 multi-level Otsu thresholds of the interior region.")
    step("3. Form the candidate mask from the interior pixels of the highest class, open it, and label its components; if there are none, return “no tumor candidate.”")
    step("4. Discard each component whose area is below A_{min} or whose clearance is below c; if none remains, return “no tumor candidate.”")
    step("5. For each remaining component, compute its contrast, inscribed radius, and shape score, rescale contrast and radius, and compute its ranking score.")
    step("6. Select the component with the highest ranking score, form S from its medial pixels, and output S.")
    step("END")
    h4("Tools")
    p("Image loading and resizing use Pillow (Clark, 2015); percentile normalization uses NumPy (Harris et al., "
      "2020). The Otsu and multi-level Otsu thresholds, disk structuring elements, morphological opening, and the "
      "region properties of the shape score come from scikit-image (van der Walt et al., 2014). Connected-"
      "component labeling and the Euclidean distance transform come from SciPy’s ndimage module (Virtanen et "
      "al., 2020). The Tkinter interface captures the clicks of the manual seeding option.")

    # ── 3.4.2 ────────────────────────────────────────────────────────────────
    h3("3.4.2 To minimize undersegmentation by replacing the global-mean difference with a local difference measure computed in the log domain, making the criterion insensitive to multiplicative bias")
    h4("Theories")
    label("The Global Homogeneity Assumption of the Original Difference Measure")
    p("The original SRG compares each candidate pixel with the mean intensity of the entire adjacent region "
      "(Adams & Bischof, 1994). A single region mean represents every pixel of the region only if the region is "
      "intensity-homogeneous. Brain tumors violate this assumption: tumor tissue is often heterogeneous, and "
      "the scanner superimposes a smooth intensity inhomogeneity on the image (Li et al., 2011). Pixels of the "
      "same tumor that lie far from the seed can then differ from the region mean more than they differ from "
      "their neighbors, so part of the tumor is left unsegmented (Akram et al., 2017).")
    label("The Multiplicative Bias-Field Model")
    p("Intensity inhomogeneity is modeled as a smooth, positive bias field that multiplies the true tissue "
      "signal (Li et al., 2011). The logarithm turns the multiplication into an addition, which is also the "
      "domain in which N4ITK models the field (Tustison et al., 2010). The model is not a result to be "
      "interpreted; it explains why the log domain is useful.")
    eq("bias", "Bias-Field Model in the Log Domain",
       ["ln I", {"rb": "x"}, " ≈ ln b", {"rb": "x"}, " + ln J", {"rb": "x"}])
    where(("*I*(*x*)", "the observed intensity of pixel x"), ("*J*(*x*)", "the true tissue signal"),
          ("*b*(*x*)", "the bias field, which varies slowly across the image"))
    p("Because the bias field varies slowly, it is nearly the same for two neighboring pixels, so it cancels when "
      "their log intensities are subtracted. The difference between a pixel and its neighbors in the log domain "
      "therefore reflects the tissue, not the bias.")
    eq("cancel", "Cancellation of the Bias Field Between Neighboring Pixels",
       ["L", {"rb": "x"}, " − L", {"rb": "y"}, " ≈ ln J", {"rb": "x"}, " − ln J", {"rb": "y"}, ",   y ∈ ", sub("W", "r"), {"rb": "x"}])
    where(("*L*(*x*) and *L*(*y*)", "the log-domain intensities of pixels x and y"),
          ("*W*_{r}(*x*)", "the small window of radius r around x, within which the bias field is nearly constant"))
    h4("Methods")
    label("Log-Domain Transformation")
    p("Every pixel of the normalized, brain-masked slice is transformed with the following equation. Its purpose "
      "is to make the multiplicative bias additive so that nearby pixels can be compared. A higher value of L "
      "means a brighter pixel; the offset ε prevents the logarithm of zero and has no practical effect on "
      "tissue intensities.")
    eq("log", "Log-Domain Transformation Formula", ["L", {"rb": "x"}, " = ln", {"rb": ["I", {"rb": "x"}, " + ε"]}])
    where(("*L*(*x*)", "the log-domain intensity of pixel x"),
          ("*I*(*x*)", "the normalized intensity from 0 to 255, equal to 0 outside the brain"),
          ("*ε*", "an offset of @v:eps@"))
    p("The log-domain image is also used to estimate the noise floor σ_{floor}, the smallest region standard "
      "deviation that the image noise allows, which the stopping criterion uses as a lower bound. It is 1.4826 "
      "times the median absolute deviation of the difference between L and its 3 × 3 median within the brain, a "
      "robust estimate of the noise that ignores the large differences at tissue edges (Donoho & Johnstone, "
      "1994; Leys et al., 2013; Rousseeuw & Croux, 1993), bounded below by @v:sigmin@ log units.")
    label("Local Log-Domain Difference Measure")
    p("The original SRG measures how different a candidate pixel is from the adjacent region with the "
      "following equation (Adams & Bischof, 1994). Its purpose is to order the candidates: the smaller the "
      "value, the more similar the pixel is to the region and the earlier it is absorbed.")
    eq("srgdiff", "Difference Measure of the Original SRG",
       [sub("δ", "SRG"), {"rb": "x"}, " = |g", {"rb": "x"}, " − ", sub("mean", "y ∈ A"), " g", {"rb": "y"}, "|"])
    where(("*δ*_{SRG}(*x*)", "the difference measure of candidate x"),
          ("*g*(*x*)", "the normalized intensity of x"), ("*A*", "the region adjacent to x"))
    p("Because the reference is the mean of the whole region, tumor pixels in a darker part of a heterogeneous "
      "tumor, or on the darker side of a bias gradient, differ from it even though they belong to the tumor. The "
      "enhanced measure instead compares x, in the log domain, with the mean log intensity of only those region "
      "pixels inside a small window around x, that is, with the pixels most recently absorbed next to it. As "
      "before, a smaller value means a more similar pixel. With r = @v:r@ the window is @v:win@ pixels, so each "
      "candidate is compared with the absorbed pixels that touch it, and by the cancellation above the bias "
      "field does not affect the comparison.")
    eq("lldm", "Local Log-Domain Difference Measure",
       ["δ", {"rb": "x"}, " = |L", {"rb": "x"}, " − ", sub("μ", "loc"), {"rb": "x"}, "|,   ", sub("μ", "loc"), {"rb": "x"}, " = ",
        {"f": ["1", ["|A ∩ ", sub("W", "r"), {"rb": "x"}, "|"]]},
        {"sum": {"lo": ["y ∈ A ∩ ", sub("W", "r"), {"rb": "x"}], "body": ["L", {"rb": "y"}]}}])
    where(("*δ*(*x*)", "the local log-domain difference measure of candidate x"),
          ("*W*_{r}(*x*)", "the (2r + 1) × (2r + 1) window centered on x"),
          ("*A* ∩ *W*_{r}(*x*)", "the set of region pixels inside that window"),
          ("*μ*_{loc}(*x*)", "their mean log intensity"))
    label("Algorithm of the Local Log-Domain Difference Measure")
    step("START")
    step("Input: the log-domain image L, the current region A, a candidate pixel x, and the window radius r. Output: δ(x).")
    step("1. Define the window of radius r centered on x, clipped at the image border.")
    step("2. From the arrays that store L and the membership of every absorbed pixel, sum the log intensities and count the region pixels in the window.")
    step("3. Compute the local mean as the sum divided by the count; if the count is zero, use the current region mean.")
    step("4. Output δ(x) as the absolute difference between L(x) and the local mean.")
    step("END")
    h4("Tools")
    p("The logarithm, absolute differences, and medians are computed with NumPy (Harris et al., 2020), and the "
      "3 × 3 median filter is SciPy’s ndimage.median_filter (Virtanen et al., 2020). The per-pixel arrays that "
      "supply the window sums are NumPy arrays updated as each pixel is absorbed.")

    # ── 3.4.3 ────────────────────────────────────────────────────────────────
    h3("3.4.3 To decrease boundary leakage by integrating a stopping criterion that adapts to the intensity variability of the growing region, supported with a global drift guard")
    h4("Theories")
    label("Unconditional Absorption in the Original Algorithm")
    p("In the SRG, the candidate pixels bordering the region are held in a sequentially sorted list (SSL) ordered "
      "by their difference measure, and the most similar candidate is always absorbed first (Adams & Bischof, "
      "1994). The original algorithm absorbs the next candidate unconditionally and stops only when every pixel "
      "has been assigned to a region. With a single tumor seed, or where the tumor boundary is weak, nothing "
      "prevents the region from crossing the boundary and flooding the surrounding tissue. Because the list is "
      "sorted, however, a bound on the difference measure can halt a growing pass exactly: once the candidate at "
      "the top of the list fails the bound, every candidate behind it fails as well.")
    label("The Rician Noise Model and Statistical Region Membership")
    p("The magnitude noise of MR images follows a Rician distribution, which approaches a Gaussian distribution "
      "when the signal-to-noise ratio exceeds about two (Gudbjartsson & Patz, 1995). The intensities of a "
      "homogeneous region can therefore be modeled as approximately normal about the region mean, so a pixel "
      "several standard deviations away is unlikely to belong to the region; even without assuming normality, "
      "Chebyshev’s inequality limits the share of values k or more standard deviations from the mean to at most "
      "1/k² (Ross, 2019). This justifies a rule that adapts to the variability of each region, the basis of "
      "confidence-connected region growing (Insight Software Consortium, n.d.).")
    label("Numerically Stable Running Statistics")
    p("The region mean and standard deviation must be updated as pixels are absorbed. Welford’s (1962) one-pass "
      "method updates both from each new value without storing the region and without the cancellation error of "
      "the sum-of-squares formula.")
    h4("Methods")
    label("Adaptive Stopping Criterion")
    p("The running mean m_{A} and standard deviation s_{A} of the log intensity are accumulated with Welford’s "
      "method from the seed onward. At the start of each pass, the region standard deviation is read and bounded "
      "below by the noise floor and above by σ_{cap}. A larger value of σ_{A} gives wider, more permissive bounds; "
      "the floor prevents a nearly uniform seed from producing bounds near zero, and the cap, @v:sigcap@ log "
      "units, prevents a heterogeneous seed from producing bounds so wide that growth floods the brain.")
    eq("sigmaA", "Region Standard Deviation Formula",
       [sub("σ", "A"), " = min", {"rb": ["max", {"rb": [sub("s", "A"), ", ", sub("σ", "floor")]}, ", ", sub("σ", "cap")]}])
    where(("*s*_{A}", "the running sample standard deviation of the region"),
          ("*σ*_{floor}", "the noise floor"), ("*σ*_{cap}", "the upper bound"),
          ("*σ*_{A}", "the region standard deviation used throughout the pass"))
    p("The two acceptance bounds of the pass are multiples of σ_{A}. The local bound limits how different a "
      "candidate may be from its neighbors, and the global bound limits how far it may stray from the region "
      "mean. A larger multiplier makes the bound more permissive.")
    eq("bounds", "Local and Global Acceptance Bounds",
       [sub("T", "L"), " = ", sub("k", "L"), sub("σ", "A"), ",   ", sub("T", "G"), " = ", sub("k", "G"), sub("σ", "A")])
    where(("*T*_{L} and *T*_{G}", "the local and global bounds"),
          ("*k*_{L} and *k*_{G}", "the local and global multipliers, @v:kL@ and @v:kG@"))
    p("A candidate y at the top of the list is absorbed only if it satisfies both conditions below. When it "
      "fails the local bound, the pass ends, because the list is sorted by δ and every remaining candidate also "
      "fails; this is the stopping criterion, and it adapts to the variability of the growing region. When it "
      "passes the local bound but fails the global bound, it is rejected and never reconsidered; this global "
      "drift guard blocks slow chaining across a soft boundary, where each pixel resembles its neighbor but the "
      "region gradually drifts away from the tumor intensity. The global test uses the running mean of the "
      "region, so the reference follows the region as it grows. The standard deviation σ_{A} stays frozen "
      "during a pass, because updating it after every absorption would let each borderline pixel widen the "
      "bounds and admit the next. Growth ends when a pass adds no pixel or after @v:pmax@ passes.")
    eq("accept", "Acceptance Condition",
       ["δ", {"rb": "y"}, " ≤ ", sub("T", "L"), "   and   |L", {"rb": "y"}, " − ", sub("m", "A"), "| ≤ ", sub("T", "G")])
    where(("*δ*(*y*)", "the local log-domain difference measure of candidate y"),
          ("*m*_{A}", "the running mean of the log intensity of the region when y is examined"))
    label("Flood Guard")
    p("A bound derived from the region’s own variability fails when the seed statistics are unrepresentative, and "
      "the region may then cover most of the brain. The flood guard detects this from the size of the region and "
      "repeats the growth with tighter bounds, up to @v:tmax@ times. Because φ = @v:phi@ is derived from the "
      "sizes of real tumors in the development data and not from the output of any method, the guard is "
      "triggered only by regions larger than almost every tumor.")
    eq("flood", "Flood Guard Rule",
       ["|A| > φ|B|  ⇒  ", {"rb": [sub("k", "L"), ", ", sub("k", "G")]}, " ← ", sup("λ", "t"), {"rb": [sub("k", "L"), ", ", sub("k", "G")]},
        ",   t = 1, …, ", sub("t", "max")])
    where(("|*A*| and |*B*|", "the areas of the grown region and of the brain, in pixels"),
          ("*φ*", "the flood fraction, @v:phi@"), ("*λ*", "the shrink factor, @v:lam@"),
          ("*t* and *t*_{max}", "the retry number and the maximum number of retries, @v:tmax@"))
    p("When the stopping criterion is applied to the baseline SRG, it acts on the tumor region only, with the "
      "same multipliers and flood guard; the SRG has no passes, so σ_{A} is read from the running statistics at "
      "each decision, and a declined pixel is left to the competing regions. In the configurations without the "
      "local measure, where the intensity is not in log units, the upper bound σ_{cap} is omitted.")
    label("Algorithm of the Adaptive Stopping Criterion")
    step("START")
    step("Input: the log-domain image, the brain mask, the seed core S, the noise floor, and the parameters of Table 3.3. Output: the grown tumor region A.")
    step("1. Set A = S and add the log intensity of every seed pixel to the running statistics.")
    step("2. For each pass, up to the maximum number of passes:")
    step("a. Read s_{A}, bound it by the noise floor and the cap to obtain σ_{A}, and set T_{L} and T_{G}.", 1)
    step("b. Start an empty SSL and insert every neighbor of A that lies in the brain and is neither in A nor rejected, with its δ.", 1)
    step("c. While the SSL is not empty, remove the pixel y with the smallest δ:", 1)
    step("i. If y is in A or rejected, skip it.", 2)
    step("ii. Recompute δ(y); if it has increased, reinsert y and continue.", 2)
    step("iii. If δ(y) > T_{L}, end the pass.", 2)
    step("iv. If the log intensity of y differs from m_{A} by more than T_{G}, reject y and continue.", 2)
    step("v. Otherwise, add y to A, update m_{A}, s_{A}, and the window arrays, and insert its eligible neighbors into the SSL.", 2)
    step("d. If no pixel was added during the pass, exit the loop.", 1)
    step("3. If A covers more than a fraction φ of the brain and fewer than t_{max} retries have been made, multiply k_{L} and k_{G} by λ, reset A = S, and return to Step 2.")
    step("4. Output A.")
    step("END")
    h4("Tools")
    p("The SSL is implemented with Python’s standard-library binary heap, heapq, which retrieves the candidate "
      "with the smallest δ in logarithmic time (Python Software Foundation, n.d.). The Welford accumulators, "
      "the acceptance tests, and the flood guard are implemented directly in Python (Welford, 1962). Connected-"
      "component labeling uses SciPy’s ndimage module (Virtanen et al., 2020).")

    h2("References Cited in This Chapter")
    p("The following works are cited in this revision and should be merged into the thesis reference list if "
      "they are not already included.", noindent=True)
    for r in REFS:
        B.append({"t": "ref", "text": r})
    return {"title": "Chapter 3: Methodology", "author": "Karylle Vinces J. Aliswag & Juana Mari A. Del Rosario", "blocks": B}


REFS = [
    "Adams, R., & Bischof, L. (1994). Seeded region growing. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 16*(6), 641–647. https://doi.org/10.1109/34.295913",
    "Baid, U., Ghodasara, S., Mohan, S., Bilello, M., Calabrese, E., Colak, E., Farahani, K., Kalpathy-Cramer, J., Kitamura, F. C., Pati, S., Prevedello, L. M., Rudie, J. D., Sako, C., Shinohara, R. T., Bergquist, T., Chai, R., Eddy, J., Elliott, J., Reade, W., … Bakas, S. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. *arXiv*. https://doi.org/10.48550/arXiv.2107.02314",
    "Bakas, S., Akbari, H., Sotiras, A., Bilello, M., Rozycki, M., Kirby, J. S., Freymann, J. B., Farahani, K., & Davatzikos, C. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. *Scientific Data, 4*, Article 170117. https://doi.org/10.1038/sdata.2017.117",
    "Biratu, E. S., Schwenker, F., Debelee, T. G., Kebede, S. R., Negera, W. G., & Molla, H. T. (2021). Enhanced region growing for brain tumor MR image segmentation. *Journal of Imaging, 7*(2), Article 22. https://doi.org/10.3390/jimaging7020022",
    "Cochran, W. G. (1977). *Sampling techniques* (3rd ed.). John Wiley & Sons.",
    "Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika, 81*(3), 425–455. https://doi.org/10.1093/biomet/81.3.425",
    "Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association, 32*(200), 675–701. https://doi.org/10.1080/01621459.1937.10503522",
    "Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. *Journal of Experimental Psychology: General, 141*(1), 2–18. https://doi.org/10.1037/a0024338",
    "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
    "Huttenlocher, D. P., Klanderman, G. A., & Rucklidge, W. J. (1993). Comparing images using the Hausdorff distance. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 15*(9), 850–863. https://doi.org/10.1109/34.232073",
    "Kendall, M. G., & Babington Smith, B. (1939). The problem of m rankings. *The Annals of Mathematical Statistics, 10*(3), 275–287. https://doi.org/10.1214/aoms/1177732186",
    "LaBella, D., Adewole, M., Alonso-Basanta, M., Altes, T., Anwar, S. M., Baid, U., Bergquist, T., Bhalerao, R., Chen, S., Chung, V., Conte, G.-M., Dako, F., Eddy, J., Ezhov, I., Godfrey, D., Hilal, F., Familiar, A., Farahani, K., Iglesias, J. E., … Calabrese, E. (2023). The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial meningioma. *arXiv*. https://doi.org/10.48550/arXiv.2305.07642",
    "Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology, 49*(4), 764–766. https://doi.org/10.1016/j.jesp.2013.03.013",
    "Lowekamp, B. C., Chen, D. T., Ibáñez, L., & Blezek, D. (2013). The design of SimpleITK. *Frontiers in Neuroinformatics, 7*, Article 45. https://doi.org/10.3389/fninf.2013.00045",
    "Maier-Hein, L., Reinke, A., Godau, P., Tizabi, M. D., Buettner, F., Christodoulou, E., Glocker, B., Isensee, F., Kleesiek, J., Kozubek, M., Reyes, M., Riegler, M. A., Wiesenfarth, M., Kavur, A. E., Sudre, C. H., Baumgartner, M., Eisenmann, M., Heckmann-Nötzel, D., Rädsch, T., … Jäger, P. F. (2024). Metrics reloaded: Recommendations for image analysis validation. *Nature Methods, 21*(2), 195–212. https://doi.org/10.1038/s41592-023-02151-z",
    "Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., Burren, Y., Porz, N., Slotboom, J., Wiest, R., Lanczi, L., Gerstner, E., Weber, M.-A., Arbel, T., Avants, B. B., Ayache, N., Buendia, P., Collins, D. L., Cordier, N., … Van Leemput, K. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE Transactions on Medical Imaging, 34*(10), 1993–2024. https://doi.org/10.1109/TMI.2014.2377694",
    "Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute deviation. *Journal of the American Statistical Association, 88*(424), 1273–1283. https://doi.org/10.1080/01621459.1993.10476408",
    "Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika, 52*(3–4), 591–611. https://doi.org/10.1093/biomet/52.3-4.591",
    "Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: Analysis, selection, and tool. *BMC Medical Imaging, 15*, Article 29. https://doi.org/10.1186/s12880-015-0068-x",
    "van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., & the scikit-image contributors. (2014). scikit-image: Image processing in Python. *PeerJ, 2*, Article e453. https://doi.org/10.7717/peerj.453",
    "Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2",
    "Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics, 4*(3), 419–420. https://doi.org/10.1080/00401706.1962.10490022",
    "Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin, 1*(6), 80–83. https://doi.org/10.2307/3001968",
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953",
    "Zijdenbos, A. P., Dawant, B. M., Margolin, R. A., & Palmer, A. C. (1994). Morphometric analysis of white matter lesions in MR images: Method and validation. *IEEE Transactions on Medical Imaging, 13*(4), 716–724. https://doi.org/10.1109/42.363096",
]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build()                      # first pass: collects the equation numbers
    spec = build()               # second pass: text and references use them
    sp = os.path.join(OUT_DIR, "chapter3_spec.json")
    json.dump(spec, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    subprocess.run(["node", os.path.join(ROOT, "experiments", "chapter4", "render_docx.js"), sp, DOCX], check=True)
    print("blocks:", len(spec["blocks"]), "| equations:", len(NUM))
