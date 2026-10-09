"""
tables.py — Every Chapter 4 table, built from results.json.

Purpose : Define each APA table of Chapter 4 once, so the Word chapter and the
          Summary sheets of the appendix workbooks show identical values.
Function : build(R) returns the full-detail tables of the appendix workbooks as
          {table id: {"num", "title", "header", "rows", "widths", "align", "note",
          "appendix"}}, numbered within each appendix; build_chapter(R) returns the
          condensed tables of the chapter itself, numbered 4.1, 4.2, …. Cells are
          display strings using the renderer's inline markup (*italic*, _{sub}, ^{sup}).
Notes   : APA 7 conventions — italic statistical symbols (M, SD, Mdn, n, N, Z, r,
          p, χ², H), no leading zero for quantities that cannot exceed 1 in
          absolute value (p, r), three decimals for overlap metrics.
"""
import math
import re

from experiments.stats import fmt_p

CLASSES = ["glioma", "meningioma", "pituitary"]
PLANES = ["axial", "coronal", "sagittal"]
LEVELS = CLASSES + ["all"]
CL = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary", "all": "All classes",
      "axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}
CFG_NAME = {
    "A_SRG": "SRG, automatic seed", "A_ESRG": "ESRG, automatic seed",
    "P_SRG": "SRG, planted seed", "P_ESRG": "ESRG (full), planted seed",
    "P_ESRG_global": "ESRG, global measure", "P_ESRG_nolog": "ESRG, log transform off",
    "P_ESRG_nostop": "ESRG, no stopping criterion", "P_ESRG_nopurify": "ESRG, unpurified click disk",
    "B_SRG": "SRG, bias field", "B_SRG_N4": "SRG + N4, bias field", "B_ESRG": "ESRG (full), bias field",
    "B_ESRG_global": "ESRG, global measure, bias field", "B_ESRG_nolog": "ESRG, log off, bias field",
}
METRIC_NAME = {"dsc": "DSC", "iou": "IoU", "precision": "Precision", "recall": "Recall",
               "hd95_wc": "HD95 (px)", "assd_wc": "ASSD (px)", "success": "Success rate (DSC ≥ 0.70)",
               "leaked": "Leakage rate (|*M*| > 2|*G*|)"}


def isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def f(v, d=3):
    return "—" if isnan(v) else f"{v:.{d}f}".replace("-", "−")


def nz(v, d=2):
    """No leading zero (APA) for statistics bounded by 1, e.g. r."""
    if isnan(v):
        return "—"
    s = f"{v:.{d}f}"
    s = s.replace("0.", ".", 1) if abs(v) < 1 else s
    return s.replace("-", "−")


def pct(v, d=1):
    return "—" if isnan(v) else f"{v:.{d}f}%"


def msd(dd, d=3):
    return "—" if not dd or not dd.get("n") else f"{f(dd['mean'], d)} ({f(dd['sd'], d)})"


def mdn_iqr(dd, d=3):
    return "—" if not dd or not dd.get("n") else f"{f(dd['median'], d)} [{f(dd['q1'], d)}, {f(dd['q3'], d)}]"


def ci(c, d=1):
    return "—" if isnan(c[0]) else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def n_(v):
    return f"{int(v):,}"


# ─── Table builders ──────────────────────────────────────────────────────────
def t_sample_size(R):
    rows = []
    for r in R["sample_size"]:
        rows.append([CL.get(r["level"], r["level"]) if r["level"] != "all" else "All classes (pooled)",
                     n_(r["N"]), f(r["n0"], 2), f(r["n_exact"], 2), n_(r["n_min"]), n_(r["n_drawn"]),
                     f"±{100 * r['margin_achieved']:.2f}%"])
    nh = R["strata"][0]["n_h"]
    return {"title": "Sample Size Required by Cochran’s Formula and Sample Size Drawn",
            "header": ["Population", "*N*", "*n*_{0}", "*n* (exact)", "*n*_{min}", "*n* drawn", "Achieved *e*"],
            "rows": rows, "widths": [2.4, 1, 1, 1.1, 1, 1, 1.2],
            "note": ("*N* = number of slices with a tumor mask in the pooled BRISC 2025 segmentation task "
                     "(training and test splits). *n*_{0} and *n* follow Cochran’s formula (Section 3.1.1) with *z* = 1.96, "
                     "*p* = .50, and *e* = .05; *n*_{min} is *n* rounded up. The largest class requirement "
                     f"({max(r['n_min'] for r in R['sample_size'] if r['level'] != 'all')} slices, pituitary) was "
                     f"applied to every class and divided equally over the three planes (Section 3.1.1), giving "
                     f"*n*_{{h}} = {nh} slices per stratum. Achieved *e* is the margin of error evaluated at the drawn sample size."),
            "appendix": "A"}


def t_strata(R):
    rows, tot = [], {"N": 0, "n": 0, "tr": 0, "te": 0}
    for s in R["strata"]:
        rows.append([CL[s["tumor"]], CL[s["plane"]], n_(s["N_h"]), n_(s["N_h_train"]), n_(s["N_h_test"]),
                     n_(s["n_h"]), n_(s["n_h_train"]), n_(s["n_h_test"]), pct(100 * s["f_h"])])
        tot["N"] += s["N_h"]; tot["n"] += s["n_h"]; tot["tr"] += s["n_h_train"]; tot["te"] += s["n_h_test"]
    rows.append(["Total", "", n_(tot["N"]), n_(sum(s["N_h_train"] for s in R["strata"])),
                 n_(sum(s["N_h_test"] for s in R["strata"])), n_(tot["n"]), n_(tot["tr"]), n_(tot["te"]),
                 pct(100 * tot["n"] / tot["N"])])
    return {"title": "Composition of the Population and of the Equal-Allocation Stratified Sample",
            "header": [[{"text": ""}, {"text": ""}, {"text": "Population (*N*_{h})", "span": 3, "rule": True},
                        {"text": "Sample (*n*_{h})", "span": 3, "rule": True}, {"text": ""}],
                       ["Tumor class", "Plane", "Total", "Train", "Test", "Total", "Train", "Test", "*f*_{h}"]],
            "rows": rows, "widths": [1.5, 1.2, 1, 1, 1, 1, 1, 1, 1],
            "note": ("Stratum *h* is a combination of tumor class and imaging plane read from each filename. "
                     "Within each stratum the slices were selected by simple random sampling without replacement "
                     "(fixed seed 2026). *f*_{h} = *n*_{h}/*N*_{h} is the sampling fraction. Train and Test give the "
                     "BRISC split each slice came from. Slice-level selection, including the random draw order of "
                     "all 4,793 slices, is listed in Appendix A."),
            "appendix": "A"}


def t_metrics():
    rows = [
        {"group": "Objective 1: automated seed selection"},
        ["Seed Hit Rate (SHR)", "Table 3.3", "Share of seeded slices whose whole seed core lies inside the tumor", "Higher",
         "Adams & Bischof (1994); Fan et al. (2005)"],
        ["Seed localization distance", "Table 3.3", "Distance from the seed centroid to the nearest tumor pixel", "Lower",
         "Fan et al. (2005)"],
        ["Within-slice SD of DSC", "—", "Spread of DSC over five simulated operator clicks on one slice", "Lower",
         "Moschidis & Graham (2010); Joskowicz et al. (2019)"],
        ["Inconsistency rate", "—", "Share of slices whose success depends on where the operator clicks", "Lower",
         "Zou et al. (2004); Visser et al. (2019)"],
        {"group": "Objective 2: undersegmentation"},
        ["Recall (sensitivity)", "Table 3.3", "Share of the true tumor recovered; 1 − recall is the missed share", "Higher",
         "Taha & Hanbury (2015); Udupa et al. (2006)"],
        {"group": "Objective 3: boundary leakage"},
        ["Precision", "Table 3.3", "Share of the predicted region that is tumor", "Higher",
         "Taha & Hanbury (2015); Maier-Hein et al. (2024)"],
        ["Leakage Rate (LR)", "Table 3.3", "Share of slices whose predicted area exceeds twice the true area", "Lower",
         "Adams & Bischof (1994); Fan et al. (2005)"],
        ["HD95", "Table 3.3", "95th percentile of the distances between the two boundaries", "Lower",
         "Huttenlocher et al. (1993); Menze et al. (2015)"],
        ["ASSD", "Table 3.3", "Mean distance between the two boundaries", "Lower",
         "Taha & Hanbury (2015); Yeghiazaryan & Voiculescu (2018)"],
        {"group": "Overall delineation and efficiency"},
        ["DSC", "Table 3.3", "Overlap of prediction and ground truth", "Higher",
         "Dice (1945); Zou et al. (2004); Menze et al. (2015)"],
        ["IoU", "Table 3.3", "Intersection divided by union", "Higher", "Jaccard (1912); Taha & Hanbury (2015)"],
        ["Success rate", "Table 3.3", "Share of slices with DSC ≥ 0.70", "Higher", "Zijdenbos et al. (1994)"],
        ["Processing time", "Table 3.3", "Wall-clock seconds per slice, sequential", "Lower",
         "Udupa et al. (2006); Hoefler & Belli (2015)"],
    ]
    return {"title": "Evaluation Metrics per Objective, Their Definitions, and Supporting Studies",
            "header": ["Metric", "Defined in", "What it measures", "Better", "Supporting studies"],
            "rows": rows, "widths": [1.9, 0.6, 3.4, 0.8, 2.6], "align": ["l", "c", "l", "c", "l"],
            "note": ("Definitions are given in Table 3.3 of Chapter 3. “Better” gives the direction in which the metric improves. All "
                     "metrics are computed per slice at the working resolution (longer side ≤ 512 pixels) and then "
                     "summarized over the sample."),
            "appendix": "F"}


def t_configs(R):
    exp = {"A_SRG": "E1, E4, E6", "A_ESRG": "E1, E4, E5, E6", "P_SRG": "E2, E3, E4", "P_ESRG": "E2, E3, E4",
           "P_ESRG_global": "E2, E4", "P_ESRG_nolog": "E2, E4", "P_ESRG_nostop": "E3, E4",
           "P_ESRG_nopurify": "E3", "B_SRG": "E2", "B_SRG_N4": "E2", "B_ESRG": "E2", "B_ESRG_global": "E2",
           "B_ESRG_nolog": "E2", "O_SRG": "E1", "O_ESRG": "E1"}
    change = {"P_ESRG_global": "Global measure, δ(*x*) = |*L*(*x*) − μ_{A}|", "P_ESRG_nolog": "*L*(*x*) = *I*(*x*)",
              "P_ESRG_nostop": "Unconditional absorption", "P_ESRG_nopurify": "Click disk not reduced to its core",
              "B_SRG_N4": "N4 correction before growing", "B_ESRG_global": "Global measure",
              "B_ESRG_nolog": "Log transform off"}
    rows = []
    for c in R["configs"]:
        k = c["id"]
        if k.startswith("O_") and not k.endswith("_1"):
            continue
        if k.startswith("O_"):
            base = k.rsplit("_", 1)[0]
            rows.append([f"{base}_1 … {base}_5", "SRG" if "SRG" in base and "ESRG" not in base else "ESRG",
                         "Operator (5 clicks)", "No", "—", exp[base]])
            continue
        rows.append([k.replace("_", "\\_") if False else k, "SRG" if c["overrides"].get("method") == "srg" else "ESRG",
                     {"auto": "Automatic", "planted": "Planted", "operator": "Operator"}[c["seeding"]],
                     "Yes" if c["biased"] else "No", change.get(k, "—"), exp[k]])
    n = R["meta"]["n_sample"]
    return {"title": "Configurations Evaluated on Every Sampled Slice",
            "header": ["Configuration", "Algorithm", "Seeding", "Bias field", "Change from full ESRG", "Experiment"],
            "rows": rows, "widths": [1.9, 0.9, 1.4, 0.8, 2.6, 1.2], "align": ["l", "c", "c", "c", "l", "c"],
            "note": (f"Every configuration was run on all {n:,} slices ({n:,} × {R['meta']['n_configs']} = "
                     f"{R['meta']['n_rows']:,} runs; {R['meta']['n_errors']} ended in an error). Automatic seeding uses "
                     "the seed selection of Objective 1; planted seeding places one click at the deepest ground-truth "
                     "pixel; operator seeding places five clicks at random tumor pixels (Section 3.1). Experiments "
                     "E1–E6 are those of Table 3.5. The identifier appears in the Configuration column of Appendix H."),
            "appendix": "H"}


def t_normality(R):
    rows = []
    for key, label in (("auto", "SRG vs. ESRG, automatic seed"), ("planted", "SRG vs. ESRG, planted seed")):
        for r in R["e4"][key]:
            if r["metric"] in ("dsc", "recall", "precision", "hd95_wc"):
                s = r["shapiro_diff"]
                rows.append([label, METRIC_NAME[r["metric"]], n_(s["n"]), nz(s["W"], 3) if s["W"] else "—",
                             fmt_p(s["p"])])
    return {"title": "Shapiro–Wilk Test of Normality of the Paired Differences",
            "header": ["Comparison", "Metric", "*N*", "*W*", "*p*"],
            "rows": rows, "widths": [3, 1.6, 0.9, 0.9, 0.9], "align": ["l", "l", "c", "c", "c"],
            "note": ("The paired difference ESRG − SRG was tested per slice. A *p* below .05 rejects normality, "
                     "so the paired *t* test is not appropriate and the Wilcoxon signed-rank test is used "
                     "(Demšar, 2006)."),
            "appendix": "F"}


def _compare_rows(rows_in, nmax):
    out = []
    for r in rows_in:
        m = r["metric"]
        if m in ("success", "leaked"):
            out.append([METRIC_NAME[m], pct(r["rate_ref"]), "—", pct(r["rate_cmp"]), "—",
                        f"{r['only_ref']} / {r['only_cmp']}", fmt_p(r["p_holm"]), "—"])
        else:
            d = 2 if m in ("hd95_wc", "assd_wc") else 3
            out.append([METRIC_NAME[m], msd(r["base"], d), f(r["base"]["median"], d), msd(r["enh"], d),
                        f(r["enh"]["median"], d), f(r["Z"], 2), fmt_p(r["p_holm"]), nz(r["r"])])
    return out


def t_overall(R, key):
    rows = _compare_rows(R["e4"][key], R["meta"]["n_sample"])
    seed = "automatic seeding" if key == "auto" else "a seed planted inside the tumor"
    n_empty = R["empty_pred"]["A_ESRG" if key == "auto" else "P_ESRG"]
    return {"title": f"Overall Performance of the Baseline and Enhanced Algorithm Under {'Automatic Seeding' if key == 'auto' else 'Planted Seeding'}",
            "header": ["Metric", "SRG *M* (*SD*)", "SRG *Mdn*", "ESRG *M* (*SD*)", "ESRG *Mdn*", "*Z* or *b* / *c*", "*p* (Holm)", "*r*"],
            "rows": rows, "widths": [2.0, 1.55, 0.85, 1.55, 0.85, 1.05, 0.85, 0.6],
            "note": (f"*N* = {R['meta']['n_sample']:,} paired slices under {seed}. *M* (*SD*) is the mean and standard "
                     "deviation over slices and *Mdn* the median. *Z* and *r* come from the Wilcoxon signed-rank test "
                     "of ESRG against SRG (Section 3.1.2); a positive value means ESRG scored higher, an "
                     "improvement for every metric except HD95 and ASSD, where lower is better. For the two rates, "
                     "*b* / *c* are the slices on which only SRG or only ESRG had the outcome (exact McNemar test, "
                     "Section 3.1.2). HD95 and ASSD of an empty prediction are set to the image diagonal "
                     f"(Table 3.3; {n_empty} ESRG slices). The eight *p* values are Holm-adjusted together."),
            "appendix": "F"}


def t_weighted(R):
    rows = []
    for c in ("A_SRG", "A_ESRG", "P_SRG", "P_ESRG"):
        for m in ("dsc", "recall", "precision"):
            u, w = R["e4"]["unweighted"][c][m], R["e4"]["weighted"][c][m]
            rows.append([CFG_NAME[c], METRIC_NAME[m], f(u["mean"]), f(w["estimate"]), f(w["se"], 4),
                         f"[{f(w['ci95'][0])}, {f(w['ci95'][1])}]"])
    return {"title": "Equal-Allocation (Sample) Means and Population-Weighted Estimates",
            "header": ["Configuration", "Metric", "Sample *M*", "Weighted *M*", "*SE*", "95% CI"],
            "rows": rows, "widths": [2.6, 1.2, 1.1, 1.2, 1, 1.6], "align": ["l", "l", "c", "c", "c", "c"],
            "note": ("Sample *M* is the unweighted mean over the equal-allocation sample. Weighted *M* re-weights "
                     "each stratum by its population share *W*_{h} = *N*_{h}/*N* (Section 3.1.1), giving an unbiased "
                     "estimate for the whole population; *SE* includes the finite population correction."),
            "appendix": "F"}


def t_seed_hit(R):
    E = R["e1"]
    rows = []
    for grp, levels, label in (("class", CLASSES + ["all"], "By tumor class"), ("plane", PLANES + ["all"], "By imaging plane")):
        rows.append({"group": label})
        for lv in levels:
            t = E[grp][lv]
            rows.append([CL[lv] if lv != "all" else "All slices", n_(t["n"]), n_(t["no_candidate"]),
                         f"{t['hits']} / {t['N_S']}", pct(t["rate"]), ci(t["ci"]),
                         mdn_iqr(t["seed_dist"], 1), f(t["inside_frac"].get("mean"), 3)])
    tc, tp = E["class"]["test"], E["plane"]["test"]
    return {"title": "Seed Hit Rate and Seed Localization of the Automated Seed Selection",
            "header": ["Group", "*n*", "No candidate", "Hits / *N*_{S}", "SHR", "95% CI", "Distance, px *Mdn* [IQR]", "Inside share *M*"],
            "rows": rows, "widths": [1.5, 0.7, 1, 1.1, 0.8, 1.2, 1.7, 1],
            "note": ("Configuration A_ESRG. A hit is a slice whose whole seed core lies inside the ground-truth tumor "
                     "(Table 3.3); *N*_{S} counts the slices that received a seed, and “No candidate” those for "
                     "which the selector reported no tumor instead of guessing. The 95% CI is the Wilson score "
                     "interval (Section 3.1.2). Distance is the seed localization distance (Table 3.3; 0 when the "
                     "centroid lies in the tumor); inside share is |*S* ∩ *G*|/|*S*|. Class differences in SHR: "
                     f"χ²({tc['df']}, *N* = {tc['n']:,}) = {tc['chi2']:.2f}, *p* {fmt_p(tc['p']).replace('< ', '< ') if fmt_p(tc['p']).startswith('<') else '= ' + fmt_p(tc['p'])}, Cramér’s *V* = {nz(tc['cramers_v'])}; "
                     f"plane differences: χ²({tp['df']}, *N* = {tp['n']:,}) = {tp['chi2']:.2f}, *p* "
                     f"{fmt_p(tp['p']) if fmt_p(tp['p']).startswith('<') else '= ' + fmt_p(tp['p'])}, *V* = {nz(tp['cramers_v'])}."),
            "appendix": "B"}


def t_hit_effect(R):
    h = R["e1"]["hit_vs_miss"]
    t = h["test"]
    rows = [["Seed hit", n_(h["hit"]["dsc"]["n"]), msd(h["hit"]["dsc"]), mdn_iqr(h["hit"]["dsc"]),
             f"{h['hit']['success']['k']} ({pct(h['hit']['success']['pct'])})"],
            ["Seed miss or no seed", n_(h["miss"]["dsc"]["n"]), msd(h["miss"]["dsc"]), mdn_iqr(h["miss"]["dsc"]),
             f"{h['miss']['success']['k']} ({pct(h['miss']['success']['pct'])})"]]
    return {"title": "Delineation Quality of the Enhanced Algorithm by Seed Outcome",
            "header": ["Seed outcome", "*n*", "DSC *M* (*SD*)", "DSC *Mdn* [IQR]", "Successful slices"],
            "rows": rows, "widths": [2, 0.8, 1.6, 2, 1.6],
            "note": ("Configuration A_ESRG. Mann–Whitney test of DSC between the two groups: "
                     f"*U* = {t['U']:,.0f}, *p* {fmt_p(t['p']) if fmt_p(t['p']).startswith('<') else '= ' + fmt_p(t['p'])}, "
                     f"rank-biserial *r* = {nz(t['r_rb'])} (Kerby, 2014)."),
            "appendix": "B"}


def t_operator(R):
    O = R["e1"]["operator"]
    rows = []
    for lv in LEVELS:
        o = O[lv]
        rows.append([CL[lv], msd(o["srg"]["sd"]), msd(o["esrg"]["sd"]), f(o["srg"]["range"]["mean"]),
                     f(o["esrg"]["range"]["mean"]),
                     pct(o["srg"]["inconsistent"]["pct"]), pct(o["esrg"]["inconsistent"]["pct"]), "0.000 (0.000)"])
    return {"title": "Variability of the Result Across Five Simulated Operators and Under Automatic Seeding",
            "header": [[{"text": ""}, {"text": "Within-slice *SD* of DSC", "span": 2, "rule": True},
                        {"text": "Mean range of DSC", "span": 2, "rule": True},
                        {"text": "Inconsistency rate", "span": 2, "rule": True}, {"text": "Automatic"}],
                       ["Tumor class", "SRG", "ESRG", "SRG", "ESRG", "SRG", "ESRG", "*SD* (both)"]],
            "rows": rows, "widths": [1.5, 1.25, 1.25, 0.85, 0.85, 1, 1, 1.25],
            "note": ("Each slice was segmented from five operator clicks placed at random tumor pixels "
                     "more than 3 pixels (the click radius) from the tumor boundary (Section 3.1; configurations O_SRG_1–5 and O_ESRG_1–5). "
                     "Within-slice *SD*, the standard deviation of the five DSC values of a slice, is shown as *M* (*SD*) over slices; the range is the largest "
                     "minus the smallest DSC of the five clicks; the inconsistency rate is the share "
                     "of slices on which some clicks succeed (DSC ≥ 0.70) and others fail. The automatic seed is "
                     "deterministic, so its within-slice *SD* is zero by construction; this was confirmed by an "
                     "independent rerun (Section 4.2.1)."),
            "appendix": "B"}


def t_buckets(R):
    E, L = R["e5"], R["e5_labels"]
    order = sorted([b for b in L if b != "?"], key=lambda b: -E["all"][b]["k"])
    rows = [[f"{b} – {L[b]}"] + [f"{E[c][b]['k']} ({pct(E[c][b]['pct'])})" for c in LEVELS] for b in order]
    rows.append(["Total"] + [n_(E[c]["n"]) for c in LEVELS])
    t = E["test"]
    return {"title": "Failure Attribution of the Enhanced Algorithm Under Automatic Seeding (Experiment E5)",
            "header": ["Bucket"] + [CL[c] for c in LEVELS],
            "rows": rows, "widths": [3.4, 1.3, 1.3, 1.3, 1.3], "align": ["l", "c", "c", "c", "c"],
            "note": ("Values are the number (and percentage) of slices in each bucket. Each slice is assigned to the "
                     "first failed gate, checked in the order of Table 3.4: X, A (head mask kept < 90% of the "
                     "tumor), B (< 30% of the tumor in the candidate mask), C (< 50% of the seed core inside the tumor), "
                     "G (final DSC ≥ 0.70), F (post-processing lowered DSC by ≥ 0.10), D (growth precision < 0.50 and "
                     "not above recall), and E (all others). Class differences in the distribution over G, B, C, and the "
                     f"remaining buckets: χ²({t['df']}, *N* = {t['n']:,}) = {t['chi2']:.2f}, *p* "
                     f"{fmt_p(t['p']) if fmt_p(t['p']).startswith('<') else '= ' + fmt_p(t['p'])}, *V* = {nz(t['cramers_v'])}."),
            "appendix": "C"}


def t_recall(R):
    E = R["e2"]["recall"]
    refs = (("P_SRG", "SRG (baseline)"), ("P_ESRG_global", "ESRG, global measure"),
            ("P_ESRG_nolog", "ESRG, log transform off"))
    rows = [{"group": "Recall *M* (*SD*)"}]
    rows += [[name] + [msd(E[lv]["tests"][k]["ref"]) for lv in LEVELS] for k, name in refs]
    rows.append(["ESRG (full)"] + [msd(E[lv]["enh"]) for lv in LEVELS])
    rows.append({"group": "Full ESRG vs. reference, *r* (*Z*)"})
    rows += [[name] + [f"{nz(E[lv]['tests'][k]['r'])} ({f(E[lv]['tests'][k]['Z'], 2)})" for lv in LEVELS]
             for k, name in refs]
    ps = [E[lv]["tests"][k]["p_holm"] for lv in LEVELS for k, _ in refs]
    return {"title": "Recall of the Baseline, the Ablated Variants, and the Full Enhanced Algorithm With a Planted Seed",
            "header": ["Configuration"] + [CL[lv] for lv in LEVELS],
            "rows": rows, "widths": [2.1, 1.3, 1.3, 1.3, 1.3], "align": ["l", "c", "c", "c", "c"],
            "note": ("Recall = TP/(TP + FN) (Table 3.3), the share of the true tumor recovered. All four "
                     "configurations start from the same planted click, so each difference is due to the growing "
                     "procedure alone. Each test is a Wilcoxon signed-rank test of full ESRG against the named "
                     "configuration; positive *Z* and *r* mean that full ESRG recovered more of the tumor. *p* values "
                     "are Holm-adjusted over the three comparisons within each class; the largest adjusted *p* is "
                     f"{fmt_p(max(ps))}."),
            "appendix": "D"}


def t_bias(R):
    B = R["e2"]["bias"]
    rows = []
    for k, name in (("B_SRG", "SRG"), ("B_ESRG_global", "ESRG, global measure"), ("B_ESRG_nolog", "ESRG, log off"),
                    ("B_ESRG", "ESRG (full)")):
        b = B[k]
        rows.append([name, msd(b["clean"]), msd(b["biased"]), f(b["biased"]["mean"] - b["clean"]["mean"]).replace("0.", "+0.", 1)
                     if b["biased"]["mean"] >= b["clean"]["mean"] else f(b["biased"]["mean"] - b["clean"]["mean"]),
                     f(b["mean_abs_change"]), f"{f(b['Z'], 2)}, {fmt_p(b['p_holm'])}",
                     f"{f(b['dsc_clean']['mean'])} → {f(b['dsc_biased']['mean'])}"])
    n4 = B["B_SRG_N4"]
    rows.append(["SRG + N4", "—", msd(n4["biased"]), "—", "—", "—", f"— → {f(n4['dsc_biased']['mean'])}"])
    return {"title": f"Recall With and Without a Synthetic {int(R['meta']['inu'] * 100)}% Bias Field (Experiment E2)",
            "header": ["Configuration", "Recall, no bias", "Recall, bias field", "Change in *M*", "*M* |change|", "*Z*, *p* (Holm)", "Mean DSC"],
            "rows": rows, "widths": [1.8, 1.3, 1.3, 1, 1, 1.3, 1.4],
            "note": ("Values are recall *M* (*SD*) over the same slices and planted seeds; only the bias field of "
                     "Section 3.1 differs. “Change in *M*” is the biased minus the unbiased mean, and *M* |change| the "
                     "average absolute per-slice change. Each configuration is compared with itself without the field "
                     "(Wilcoxon signed-rank test, Holm-adjusted over the four configurations). N4 was applied only "
                     f"under the field; against plain SRG on the biased slices it changed recall with *Z* = "
                     f"{f(n4['vs_B_SRG']['Z'], 2)}, *p* {fmt_p(n4['vs_B_SRG']['p']) if fmt_p(n4['vs_B_SRG']['p']).startswith('<') else '= ' + fmt_p(n4['vs_B_SRG']['p'])}."),
            "appendix": "D"}


def t_leak(R):
    E = R["e3"]
    rows = []
    for lv in LEVELS:
        p = E["precision"][lv]
        lk = E["leak"][lv]
        rows.append([CL[lv], msd(p["tests"]["P_ESRG_nostop"]["ref"]), msd(p["enh"]),
                     f"{nz(p['tests']['P_ESRG_nostop']['r'])} ({f(p['tests']['P_ESRG_nostop']['Z'], 2)})",
                     pct(lk["leak_ci"]["P_ESRG_nostop"]["pct"]), pct(lk["leak_ci"]["P_ESRG"]["pct"]),
                     f"{lk['P_ESRG_nostop']['only_ref']} / {lk['P_ESRG_nostop']['only_cmp']}",
                     f"{f(lk['ratio']['P_ESRG_nostop']['median'], 2)} → {f(lk['ratio']['P_ESRG']['median'], 2)}"])
    return {"title": "Precision and Leakage With and Without the Adaptive Stopping Criterion",
            "header": [[{"text": ""}, {"text": "Precision *M* (*SD*)", "span": 3, "rule": True},
                        {"text": "Leakage rate", "span": 3, "rule": True}, {"text": ""}],
                       ["Tumor class", "No stopping", "Adaptive", "*r* (*Z*)", "No stopping", "Adaptive", "*b* / *c*", "|*M*|/|*G*| *Mdn*"]],
            "rows": rows, "widths": [1.3, 1.2, 1.2, 1.0, 1.05, 1.05, 0.8, 1.3],
            "note": ("Both configurations use the local log-domain measure and the same planted seed; only the "
                     "stopping criterion differs. Precision = TP/(TP + FP) (Table 3.3), tested with the Wilcoxon "
                     "signed-rank test, Holm-adjusted over the two reference configurations (no stopping criterion and SRG) within each row. Leakage "
                     "is a predicted area more than twice the ground-truth area (Table 3.3); *b* / *c* are the slices "
                     "that leaked only without or only with the criterion (exact McNemar test). The last column is the "
                     "median area ratio without → with the criterion."),
            "appendix": "E"}


def t_boundary(R):
    E = R["e3"]
    rows = []
    for col in ("hd95_wc", "assd_wc"):
        rows.append({"group": METRIC_NAME[col]})
        for lv in LEVELS:
            e = E[col][lv]
            t = e["tests"]
            rows.append([CL[lv], mdn_iqr(t["P_SRG"]["ref"], 1), mdn_iqr(t["P_ESRG_nostop"]["ref"], 1), mdn_iqr(e["enh"], 1),
                         f(t['P_SRG']['Z'], 2), f(t['P_ESRG_nostop']['Z'], 2)])
    pmax = max(E[c][lv]["tests"][k]["p_holm"] for c in ("hd95_wc", "assd_wc") for lv in LEVELS
               for k in ("P_SRG", "P_ESRG_nostop"))
    return {"title": "Boundary Distances of the Baseline, the Enhanced Algorithm Without Stopping, and the Full Enhanced Algorithm",
            "header": [[{"text": ""}, {"text": "*Mdn* [IQR], pixels", "span": 3, "rule": True},
                        {"text": "Full ESRG vs. … *Z*", "span": 2, "rule": True}],
                       ["Tumor class", "SRG", "ESRG, no stopping", "ESRG (full)", "SRG", "No stop"]],
            "rows": rows, "widths": [0.9, 1.4, 1.4, 1.4, 0.65, 0.75],
            "note": ("Planted seed. HD95 (Table 3.3) and ASSD (Table 3.3) in pixels at the working resolution; "
                     "lower is better, so a negative *Z* means that full ESRG had the smaller distance on most slices. "
                     "*Z* is the Wilcoxon signed-rank statistic; after Holm adjustment over the two comparisons within "
                     f"each row, every *p* is {fmt_p(pmax) if fmt_p(pmax).startswith('<') else '≤ ' + fmt_p(pmax)}."),
            "appendix": "E"}


def t_purify(R):
    P = R["e3"]["purify"]
    rows = _compare_rows(P["table"], R["meta"]["n_sample"])
    rows = [[r[0], r[1], r[3], r[5], r[6], r[7]] for r in rows]
    rows += [["Seed area, px (*Mdn*)", f(P["seed_area"]["P_ESRG_nopurify"]["median"], 0), f(P["seed_area"]["P_ESRG"]["median"], 0), "—", "—", "—"],
             ["Initial σ_{A} (*Mdn*)", f(P["sigma_A"]["P_ESRG_nopurify"]["median"], 4), f(P["sigma_A"]["P_ESRG"]["median"], 4), "—", "—", "—"],
             ["σ_{A} set by σ_{floor}", pct(P["floor_binding"]["P_ESRG_nopurify"]), pct(P["floor_binding"]["P_ESRG"]), "—", "—", "—"]]
    return {"title": "Effect of Seed Purification on the Enhanced Algorithm With a Planted Seed",
            "header": ["Metric", "Click disk *M* (*SD*)", "Purified core *M* (*SD*)", "*Z* or *b* / *c*", "*p* (Holm)", "*r*"],
            "rows": rows, "widths": [2.2, 1.6, 1.6, 1.2, 1, 0.7],
            "note": ("Configurations P_ESRG_nopurify (the click disk used directly) and P_ESRG (the disk reduced to its "
                     "core, the default in manual mode). Tests compare the purified core against the disk; *p* values "
                     "are Holm-adjusted together. Initial σ_{A} is max(*s*_{A}, σ_{floor}) in the first pass (Equation 3.9); "
                     "the last row is the share of slices on which the noise floor, not the seed, set that value."),
            "appendix": "E"}


def t_dsc_class(R):
    B = R["e4"]["by_class"]
    rows = []
    for lv in LEVELS:
        b = B[lv]
        rows.append([CL[lv],
                     f"{f(b['A_SRG']['dsc']['median'])} ({pct(b['A_SRG']['success']['pct'])})",
                     f"{f(b['A_ESRG']['dsc']['median'])} ({pct(b['A_ESRG']['success']['pct'])})",
                     nz(b["auto_test"]["r"]),
                     f"{f(b['P_SRG']['dsc']['median'])} ({pct(b['P_SRG']['success']['pct'])})",
                     f"{f(b['P_ESRG']['dsc']['median'])} ({pct(b['P_ESRG']['success']['pct'])})",
                     nz(b["planted_test"]["r"])])
    k = R["e4"]["kruskal"]
    pmax = max(B[lv][t]["p_holm"] for lv in LEVELS for t in ("auto_test", "planted_test"))
    return {"title": "Dice Similarity Coefficient by Tumor Class",
            "header": [[{"text": ""}, {"text": "Automatic seed", "span": 3, "rule": True},
                        {"text": "Planted seed", "span": 3, "rule": True}],
                       ["Tumor class", "SRG", "ESRG", "*r*", "SRG", "ESRG", "*r*"]],
            "rows": rows, "widths": [1.3, 1.3, 1.3, 0.55, 1.3, 1.3, 0.55],
            "note": ("Values are the median DSC with the success rate (DSC ≥ 0.70) in parentheses. *r* = *Z*/√*N* "
                     "(Section 3.1.2) is the effect size of the Wilcoxon signed-rank test of ESRG against SRG under the "
                     "same seeding; after Holm adjustment over the three classes every *p* is "
                     f"{fmt_p(pmax) if fmt_p(pmax).startswith('<') else '≤ ' + fmt_p(pmax)}. DSC of ESRG differs across classes under automatic "
                     f"seeding, *H*({k['A_ESRG']['class']['df']}) = {k['A_ESRG']['class']['H']:.2f}, *p* "
                     f"{fmt_p(k['A_ESRG']['class']['p']) if fmt_p(k['A_ESRG']['class']['p']).startswith('<') else '= ' + fmt_p(k['A_ESRG']['class']['p'])}, "
                     f"and planted seeding, *H*({k['P_ESRG']['class']['df']}) = {k['P_ESRG']['class']['H']:.2f}, *p* "
                     f"{fmt_p(k['P_ESRG']['class']['p']) if fmt_p(k['P_ESRG']['class']['p']).startswith('<') else '= ' + fmt_p(k['P_ESRG']['class']['p'])} (Kruskal–Wallis)."),
            "appendix": "F"}


def t_dsc_plane(R):
    G = R["e4"]["class_plane"]
    rows = []
    for c in LEVELS:
        rows.append([CL[c]] + [f"{f(G[f'{c}|{p}'][k]['dsc']['median'])} ({pct(G[f'{c}|{p}'][k]['success']['pct'])})"
                               for k in ("A_ESRG", "P_ESRG") for p in PLANES])
    k = R["e4"]["kruskal"]
    pa, pp = k["A_ESRG"]["plane"], k["P_ESRG"]["plane"]
    return {"title": "Median DSC of the Enhanced Algorithm by Tumor Class and Imaging Plane",
            "header": [[{"text": ""}, {"text": "Automatic seed", "span": 3, "rule": True},
                        {"text": "Planted seed", "span": 3, "rule": True}],
                       ["Tumor class", "Axial", "Coronal", "Sagittal", "Axial", "Coronal", "Sagittal"]],
            "rows": rows, "widths": [1.4, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3],
            "note": ("Values are the median DSC of ESRG with the success rate in parentheses; every class × plane cell "
                     f"holds *n*_{{h}} = {R['strata'][0]['n_h']} slices. Across planes, DSC differs under automatic seeding "
                     f"with *H*({pa['df']}) = {pa['H']:.2f}, *p* {fmt_p(pa['p']) if fmt_p(pa['p']).startswith('<') else '= ' + fmt_p(pa['p'])}, "
                     f"and under planted seeding with *H*({pp['df']}) = {pp['H']:.2f}, *p* "
                     f"{fmt_p(pp['p']) if fmt_p(pp['p']).startswith('<') else '= ' + fmt_p(pp['p'])} (Kruskal–Wallis)."),
            "appendix": "F"}


def t_ablation(R):
    A = R["e4"]["ablation"]
    S = A["summary"]
    rows = []
    for k in ("P_ESRG", "P_ESRG_global", "P_ESRG_nolog", "P_ESRG_nostop", "P_SRG"):
        s = S[k]
        pw = A["pairwise"].get(k)
        rows.append([{"P_ESRG": "ESRG (full)", "P_ESRG_global": "Global measure", "P_ESRG_nolog": "Log transform off",
                      "P_ESRG_nostop": "No stopping criterion", "P_SRG": "SRG (baseline)"}[k],
                     f(s["dsc"]["mean"]), f(s["recall"]["mean"]), f(s["precision"]["mean"]),
                     pct(s["success"]["pct"]), pct(s["leaked"]["pct"]),
                     f(A["mean_rank"][k], 2) if k in A["mean_rank"] else "—",
                     f"{f(pw['Z'], 2)}, {fmt_p(pw['p_holm'])}" if pw else "—"])
    return {"title": "Contribution of Each Enhancement With a Planted Seed (Experiment E4)",
            "header": ["Configuration", "DSC *M*", "Recall *M*", "Precision *M*", "Success", "Leakage", "Mean rank", "vs. full *Z*, *p*"],
            "rows": rows, "widths": [2, 0.9, 0.9, 1, 0.9, 0.9, 0.9, 1.4],
            "note": ("All rows use the same planted seeds. Each ablated configuration removes exactly one enhancement "
                     "from full ESRG. Mean rank is the average rank of the configuration’s DSC among the four ESRG "
                     f"configurations on each slice (1 = best) from the Friedman test (Section 3.1.2): χ²({A['df']}, "
                     f"*N* = {A['n']:,}) = {A['chi2']:.2f}, *p* {fmt_p(A['p']) if fmt_p(A['p']).startswith('<') else '= ' + fmt_p(A['p'])}, "
                     f"Kendall’s *W* = {nz(A['kendall_w'])}. The pairwise *Z* compares the ablated configuration with "
                     "full ESRG on DSC (positive = full ESRG higher), Holm-adjusted over the three comparisons. "
                     "The baseline is listed for reference and is not part of the Friedman test."),
            "appendix": "F"}


def t_time(R):
    T = R["e6"]
    rows = []
    for lv in LEVELS:
        t = T[lv]
        rows.append([CL[lv], msd(t["srg"], 2), f(t["srg"]["median"], 2), msd(t["esrg"], 2), f(t["esrg"]["median"], 2),
                     pct(t["reduction_pct_median"]), f(t["Z"], 2), nz(t["r"])])
    pmax = max(T[lv]["p_holm"] for lv in LEVELS)
    env = R["meta"].get("environment") or {}
    return {"title": "Processing Time per Slice Under Automatic Seeding (Experiment E6)",
            "header": ["Tumor class", "SRG *M* (*SD*), s", "SRG *Mdn*, s", "ESRG *M* (*SD*), s", "ESRG *Mdn*, s",
                       "Reduction (*Mdn*)", "*Z*", "*r*"],
            "rows": rows, "widths": [1.2, 1.35, 0.8, 1.35, 0.8, 0.95, 0.7, 0.5],
            "note": (f"*N* = {T['all']['N']} slices, a stratified random subsample of {T['all']['N'] // 9} slices per "
                     "stratum. Wall-clock time from loading the slice to the final mask, excluding the computation of "
                     "the metrics, measured in a separate sequential pass (one process, nothing else running, one "
                     "discarded warm-up run, order of SRG and ESRG alternated per slice) on "
                     f"{env.get('cpu', 'the hardware of Table 3.1')} with Python {env.get('python', '')}"
                     + (" on battery power" if env.get("source") == "battery" else "") + ". Reduction is "
                     "1 − *Mdn*_{ESRG}/*Mdn*_{SRG}. A negative *Z* means ESRG was faster (Wilcoxon signed-rank test); after "
                     "Holm adjustment over the three classes every *p* is "
                     f"{fmt_p(pmax) if fmt_p(pmax).startswith('<') else '≤ ' + fmt_p(pmax)}."),
            "appendix": "G"}


def t_stages(R):
    S = R["e6"]["stages"]
    names = [("t_input", "1 · Input"), ("t_mask", "2 · Head mask"), ("t_log", "3 · Log domain"),
             ("t_candidates", "4–5 · Seed selection"), ("t_growth", "6 · Region growing"), ("t_final", "7 · Post-processing")]
    rows = [[n, msd(S["A_SRG"][k], 3), f(S["A_SRG"][k]["median"], 3), msd(S["A_ESRG"][k], 3), f(S["A_ESRG"][k]["median"], 3)]
            for k, n in names]
    return {"title": "Processing Time per Pipeline Stage",
            "header": ["Stage", "SRG *M* (*SD*), s", "SRG *Mdn*, s", "ESRG *M* (*SD*), s", "ESRG *Mdn*, s"],
            "rows": rows, "widths": [2.2, 1.6, 1.1, 1.6, 1.1], "align": ["l", "c", "c", "c", "c"],
            "note": ("Stage times from the same sequential pass as the per-slice processing times. Stages 1–5 are shared by both "
                     "algorithms, so their times differ only by measurement noise; the difference in total time comes "
                     "from Stage 6. The remainder of the total (noise-floor estimation and bookkeeping) is not shown."),
            "appendix": "G"}


ORDER = [("sample_size", t_sample_size), ("strata", t_strata), ("configs", t_configs), ("metrics", None),
         ("normality", t_normality), ("overall_auto", lambda R: t_overall(R, "auto")),
         ("overall_planted", lambda R: t_overall(R, "planted")), ("weighted", t_weighted),
         ("seed_hit", t_seed_hit), ("hit_effect", t_hit_effect), ("operator", t_operator), ("buckets", t_buckets),
         ("recall", t_recall), ("bias", t_bias), ("leak", t_leak), ("boundary", t_boundary), ("purify", t_purify),
         ("dsc_class", t_dsc_class), ("dsc_plane", t_dsc_plane), ("ablation", t_ablation), ("time", t_time),
         ("stages", t_stages)]


def build(R):
    """Full-detail tables for the appendix workbooks, numbered within each appendix (B.1, B.2, …)."""
    out, count = {}, {}
    for key, fn in ORDER:
        t = t_metrics() if key == "metrics" else fn(R)
        count[t["appendix"]] = count.get(t["appendix"], 0) + 1
        t["num"] = f"{t['appendix']}.{count[t['appendix']]}"
        out[key] = t
    return out


# ─── Chapter 4 tables ────────────────────────────────────────────────────────
# Condensed versions for the chapter itself: means and medians without SD, the
# effect size r instead of Z, and p only where it is not < .001. The full-detail
# tables above stay in the appendix workbooks. A note may reference another
# chapter table as [[T:key]]; build_chapter() fills in its number.
ALPHA = 0.05


def _eff(r):
    a = abs(r or 0)
    return "large" if a >= 0.5 else "medium" if a >= 0.3 else "small" if a >= 0.1 else "negligible"


def verdict(p, r=None, favors=None):
    """Interpretation cell: 'Significant, large effect; favors ESRG' or 'Not significant' (Holm-adjusted p < .05)."""
    if p is None:
        return "—"
    if p >= ALPHA:
        return "Not significant"
    out = "Significant"
    if r is not None:
        out += f", {_eff(r)} effect"
    return out + (f"; favors {favors}" if favors else "")


def _p_all(ps):
    """'every *p* is < .001' or 'every *p* is ≤ .023'."""
    s = fmt_p(max(ps))
    return f"every *p* is {s}" if s.startswith("<") else f"every *p* is ≤ {s}"


def c_overall(R, key):
    rows = []
    for r in R["e4"][key]:
        m = r["metric"]
        if m in ("iou", "assd_wc"):
            continue
        if m in ("success", "leaked"):
            rows.append([METRIC_NAME[m], pct(r["rate_ref"]), "—", pct(r["rate_cmp"]), "—", fmt_p(r["p_holm"]), "—"])
        else:
            d = 2 if m == "hd95_wc" else 3
            rows.append([METRIC_NAME[m], f(r["base"]["mean"], d), f(r["base"]["median"], d), f(r["enh"]["mean"], d),
                         f(r["enh"]["median"], d), fmt_p(r["p_holm"]), nz(r["r"])])
    n_empty = R["empty_pred"]["A_ESRG" if key == "auto" else "P_ESRG"]
    seed = "automatic seeding" if key == "auto" else "a seed planted inside the tumor"
    return {"title": f"Overall Performance of the Baseline and Enhanced Algorithm Under {'Automatic' if key == 'auto' else 'Planted'} Seeding",
            "header": [[{"text": ""}, {"text": "SRG", "span": 2, "rule": True}, {"text": "ESRG", "span": 2, "rule": True},
                        {"text": ""}, {"text": ""}],
                       ["Metric", "*M*", "*Mdn*", "*M*", "*Mdn*", "*p*", "*r*"]],
            "rows": rows, "widths": [2.6, 1, 1, 1, 1, 0.9, 0.7],
            "note": (f"*N* = {R['meta']['n_sample']:,} paired slices under {seed}. *p* is from the Wilcoxon signed-rank "
                     "test, or the exact McNemar test for the two rates, Holm-adjusted over the metrics of the comparison; "
                     "*r* = *Z*/√*N* (Section 3.1.2) is positive when ESRG scored higher, an improvement for every metric "
                     "except HD95, where lower is better. HD95 of an empty prediction is set to the image diagonal "
                     f"(Table 3.3; {n_empty} ESRG slices). IoU, ASSD, standard deviations, and test statistics are "
                     "given in Appendix F.")}


def c_seed_hit(R):
    E = R["e1"]
    rows = []
    for grp, levels, label in (("class", CLASSES + ["all"], "By tumor class"), ("plane", PLANES, "By imaging plane")):
        rows.append({"group": label})
        for lv in levels:
            t = E[grp][lv]
            last = lv == levels[-1]
            tst = E[grp]["test"]
            interp = (f"{'Class' if grp == 'class' else 'Plane'} differences: {verdict(tst['p'])}") if last else ""
            rows.append([CL[lv] if lv != "all" else "All slices", n_(t["no_candidate"]), f"{t['hits']} / {t['N_S']}",
                         pct(t["rate"]), ci(t["ci"]), f(t["seed_dist"]["median"], 1), interp])
    tc, tp = E["class"]["test"], E["plane"]["test"]
    return {"title": "Seed Hit Rate of the Automated Seed Selection",
            "header": ["Group", "No candidate", "Hits / *N*_{S}", "SHR", "95% CI", "Distance *Mdn*, px", "Interpretation"],
            "rows": rows, "widths": [1.5, 1, 1.1, 0.8, 1.2, 1.2, 1.7],
            "note": (f"Configuration A_ESRG; each class and each plane holds {E['class']['glioma']['n']} slices. A hit is a "
                     "slice whose whole seed core lies inside the tumor (Table 3.3); *N*_{S} counts the slices that "
                     "received a seed, and “No candidate” those for which the selector reported no tumor instead of "
                     "guessing. The 95% CI is the Wilson score interval (Section 3.1.2). Distance is the seed distance "
                     "(Table 3.3), 0 when the seed centroid lies inside the tumor. Class differences: "
                     f"χ²({tc['df']}, *N* = {tc['n']:,}) = {tc['chi2']:.2f}, *p* {_pv(tc['p'])}, Cramér’s *V* = "
                     f"{nz(tc['cramers_v'])}; plane differences: χ²({tp['df']}, *N* = {tp['n']:,}) = {tp['chi2']:.2f}, "
                     f"*p* {_pv(tp['p'])}, *V* = {nz(tp['cramers_v'])}.")}


def _pv(p):
    """'< .001' or '= .023', for use after an italic *p*."""
    s = fmt_p(p)
    return s if s.startswith("<") else f"= {s}"


def c_operator(R):
    O = R["e1"]["operator"]
    rows = [[CL[lv], f(O[lv]["srg"]["sd"]["mean"]), f(O[lv]["esrg"]["sd"]["mean"]),
             pct(O[lv]["srg"]["inconsistent"]["pct"]), pct(O[lv]["esrg"]["inconsistent"]["pct"]),
             verdict(O[lv]["sd_test"]["p"], O[lv]["sd_test"]["r"],
                     "SRG (smaller SD)" if O[lv]["sd_test"]["Z"] > 0 else "ESRG")] for lv in LEVELS]
    return {"title": "Variability of the Result Across Five Simulated Operators",
            "header": [[{"text": ""}, {"text": "Within-slice *SD* of DSC", "span": 2, "rule": True},
                        {"text": "Inconsistency rate", "span": 2, "rule": True}, {"text": ""}],
                       ["Tumor class", "SRG", "ESRG", "SRG", "ESRG", "Interpretation of the *SD*"]],
            "rows": rows, "widths": [1.5, 1, 1, 1, 1, 2.1],
            "note": ("Each slice was segmented from five simulated operator clicks at random tumor pixels (Section 3.1). "
                     "The within-slice *SD* is the standard deviation of the five DSC values of a slice, averaged over "
                     "slices; the inconsistency rate is the share of slices on which some clicks succeed (DSC ≥ 0.70) and "
                     "others fail. The automatic seed is deterministic, so both measures are zero under automatic seeding.")}


def c_buckets(R):
    t = dict(t_buckets(R))
    s = R["e5"]["test"]
    t["note"] = ("Number (and percentage) of slices in each bucket. Each slice is assigned to the first gate it fails, in "
                 "the order of Table 3.4. Class differences in the distribution over G, B, C, and the remaining buckets: "
                 f"χ²({s['df']}, *N* = {s['n']:,}) = {s['chi2']:.2f}, *p* {_pv(s['p'])}, *V* = {nz(s['cramers_v'])}.")
    return t


def _recall_verdict(E, k):
    """Verdict of full ESRG against reference k on all slices, flagging any class that is not significant."""
    a = E["all"]["tests"][k]
    text = verdict(a["p_holm"], a["r"], "full ESRG" if a["r"] > 0 else "the reference")
    odd = [CL[lv].lower() for lv in CLASSES if E[lv]["tests"][k]["p_holm"] >= ALPHA]
    return text + (f" (not in {', '.join(odd)})" if odd else "")


def c_recall(R):
    E = R["e2"]["recall"]
    refs = (("P_SRG", "SRG (baseline)"), ("P_ESRG_global", "ESRG, global measure"),
            ("P_ESRG_nolog", "ESRG, log transform off"))
    rows = [{"group": "Recall *M*"}]
    rows += [[name] + [f(E[lv]["tests"][k]["ref"]["mean"]) for lv in LEVELS] + [""] for k, name in refs]
    rows.append(["ESRG (full)"] + [f(E[lv]["enh"]["mean"]) for lv in LEVELS] + [""])
    rows.append({"group": "Full ESRG vs. reference, *r*"})
    rows += [[name] + [nz(E[lv]["tests"][k]["r"]) for lv in LEVELS]
             + [_recall_verdict(E, k)] for k, name in refs]
    ps = [E[lv]["tests"][k]["p_holm"] for lv in LEVELS for k, _ in refs]
    return {"title": "Recall of the Baseline, the Ablated Variants, and the Full Enhanced Algorithm With a Planted Seed",
            "header": ["Configuration"] + [CL[lv] for lv in LEVELS] + ["Interpretation"],
            "rows": rows, "widths": [1.9, 1, 1.25, 1, 1, 1.9], "align": ["l", "c", "c", "c", "c", "l"],
            "note": ("Recall (Table 3.3) is the share of the true tumor recovered. All four configurations start from "
                     "the same planted click, so each difference is due to the growing procedure alone. *r* is the effect "
                     "size of the Wilcoxon signed-rank test of full ESRG against the named configuration (positive = full "
                     f"ESRG recovered more); after Holm adjustment over the three comparisons within each class, {_p_all(ps)}.")}


def c_bias(R):
    B = R["e2"]["bias"]
    rows = []
    for k, name in (("B_SRG", "SRG"), ("B_ESRG_global", "ESRG, global measure"), ("B_ESRG_nolog", "ESRG, log off"),
                    ("B_ESRG", "ESRG (full)")):
        b = B[k]
        dlt = b["biased"]["mean"] - b["clean"]["mean"]
        rows.append([name, f(b["clean"]["mean"]), f(b["biased"]["mean"]), ("+" if dlt >= 0 else "") + f(dlt),
                     fmt_p(b["p_holm"]),
                     verdict(b["p_holm"], b["r"]) + ("" if b["p_holm"] >= ALPHA else
                             ("; recall higher with the field" if b["r"] > 0 else "; recall lower with the field"))])
    n4 = B["B_SRG_N4"]
    rows.append(["SRG + N4", "—", f(n4["biased"]["mean"]), "—", "—", "—"])
    return {"title": f"Recall With and Without a Synthetic {int(R['meta']['inu'] * 100)}% Bias Field",
            "header": ["Configuration", "Recall *M*, no bias", "Recall *M*, bias field", "Change", "*p*", "Interpretation"],
            "rows": rows, "widths": [1.9, 1.2, 1.2, 0.9, 0.8, 2],
            "note": ("Same slices and planted seeds; only the bias field of Section 3.1 differs. Each "
                     "configuration is compared with itself without the field (Wilcoxon signed-rank test, Holm-adjusted "
                     "over the four configurations). N4 was applied only under the field; against plain SRG on the "
                     f"biased slices, *p* {_pv(n4['vs_B_SRG']['p'])}.")}


def c_leak(R):
    E = R["e3"]
    rows, ps = [], []
    for lv in LEVELS:
        p, lk = E["precision"][lv], E["leak"][lv]
        rows.append([CL[lv], f(p["tests"]["P_ESRG_nostop"]["ref"]["mean"]), f(p["enh"]["mean"]),
                     nz(p["tests"]["P_ESRG_nostop"]["r"]),
                     pct(lk["leak_ci"]["P_ESRG_nostop"]["pct"]), pct(lk["leak_ci"]["P_ESRG"]["pct"]),
                     "Precision: " + verdict(p["tests"]["P_ESRG_nostop"]["p_holm"], p["tests"]["P_ESRG_nostop"]["r"]) +
                     "; leakage: " + verdict(lk["P_ESRG_nostop"]["p"])])
        ps += [p["tests"]["P_ESRG_nostop"]["p_holm"], lk["P_ESRG_nostop"]["p"]]
    return {"title": "Precision and Leakage Rate With and Without the Adaptive Stopping Criterion",
            "header": [[{"text": ""}, {"text": "Precision *M*", "span": 3, "rule": True},
                        {"text": "Leakage rate", "span": 2, "rule": True}, {"text": ""}],
                       ["Tumor class", "No stopping", "Adaptive", "*r*", "No stopping", "Adaptive", "Interpretation"]],
            "rows": rows, "widths": [1.3, 1, 1, 0.6, 1, 1, 2.6],
            "note": ("Both configurations use the local log-domain measure and the same planted seed; only the stopping "
                     "criterion differs. Precision (Table 3.3) was compared with the Wilcoxon signed-rank test and the "
                     "leakage rate (Table 3.3; predicted area more than twice the true area) with the exact McNemar "
                     f"test; {_p_all(ps)}.")}


def _cap(t):
    return t[:1].upper() + t[1:]


def _hd_verdict(t, ref):
    """HD95: lower is better, so a positive Z (full ESRG larger) favors the reference."""
    if t["p_holm"] >= ALPHA:
        return f"not significant vs. {ref}"
    return f"significant vs. {ref} ({_eff(t['r'])} effect, favors {'full ESRG' if t['Z'] < 0 else ref})"


def c_boundary(R):
    E = R["e3"]["hd95_wc"]
    rows = [[CL[lv], mdn_iqr(E[lv]["tests"]["P_SRG"]["ref"], 1), mdn_iqr(E[lv]["tests"]["P_ESRG_nostop"]["ref"], 1),
             mdn_iqr(E[lv]["enh"], 1),
             _cap(_hd_verdict(E[lv]["tests"]["P_SRG"], "SRG") + "; "
                  + _hd_verdict(E[lv]["tests"]["P_ESRG_nostop"], "no stopping"))] for lv in LEVELS]
    ps = [E[lv]["tests"][k]["p_holm"] for lv in LEVELS for k in ("P_SRG", "P_ESRG_nostop")]
    worse = [CL[lv].lower() for lv in CLASSES if E[lv]["tests"]["P_SRG"]["Z"] > 0]
    return {"title": "HD95 of the Baseline, the Enhanced Algorithm Without Stopping, and the Full Enhanced Algorithm",
            "header": [[{"text": ""}, {"text": "HD95 *Mdn* [IQR], pixels", "span": 3, "rule": True}, {"text": ""}],
                       ["Tumor class", "SRG", "ESRG, no stopping", "ESRG (full)", "Interpretation"]],
            "rows": rows, "widths": [1.2, 1.3, 1.4, 1.3, 3.2],
            "note": ("Planted seed. HD95 (Table 3.3) in pixels at the working resolution; lower is better. Full ESRG "
                     "was compared with each reference by the Wilcoxon signed-rank test, Holm-adjusted within each row; "
                     f"{_p_all(ps)}. Full ESRG has the smaller HD95 against both references in every class"
                     + (f" except against SRG for {', '.join(worse)}" if worse else "") + ". ASSD gives the same pattern "
                     "(Appendix E).")}


def c_purify(R):
    P = {r["metric"]: r for r in R["e3"]["purify"]["table"]}
    rows, ps = [], []
    for m in ("dsc", "recall", "precision", "success", "leaked"):
        r = P[m]
        ps.append(r["p_holm"])
        if m in ("success", "leaked"):
            better = (r["rate_cmp"] > r["rate_ref"]) == (m == "success")
            rows.append([METRIC_NAME[m], pct(r["rate_ref"]), pct(r["rate_cmp"]), "—",
                         verdict(r["p_holm"], None, "the purified core" if better else "the click disk")])
        else:
            rows.append([METRIC_NAME[m], f(r["base"]["mean"]), f(r["enh"]["mean"]), nz(r["r"]),
                         verdict(r["p_holm"], r["r"], "the purified core" if r["r"] > 0 else "the click disk")])
    return {"title": "Effect of Seed Purification on the Enhanced Algorithm With a Planted Seed",
            "header": ["Metric", "Click disk", "Purified core", "*r*", "Interpretation"],
            "rows": rows, "widths": [2.1, 1.1, 1.1, 0.7, 2.7],
            "note": ("Means and rates of P_ESRG_nopurify (the click disk used directly) and P_ESRG (the disk reduced to its "
                     "core, the default in manual mode). *r* is positive when the purified core scored higher (Wilcoxon "
                     f"signed-rank test; rates: exact McNemar test); after Holm adjustment, {_p_all(ps)}.")}


def c_ablation(R):
    A = R["e4"]["ablation"]
    S = A["summary"]
    rows = []
    for k in ("P_ESRG", "P_ESRG_global", "P_ESRG_nolog", "P_ESRG_nostop", "P_SRG"):
        s, pw = S[k], A["pairwise"].get(k)
        rows.append([{"P_ESRG": "ESRG (full)", "P_ESRG_global": "Global measure", "P_ESRG_nolog": "Log transform off",
                      "P_ESRG_nostop": "No stopping criterion", "P_SRG": "SRG (baseline)"}[k],
                     f(s["dsc"]["mean"]), pct(s["success"]["pct"]), pct(s["leaked"]["pct"]),
                     f(A["mean_rank"][k], 2) if k in A["mean_rank"] else "—", nz(pw["r"]) if pw else "—"])
    ps = [v["p_holm"] for v in A["pairwise"].values()]
    return {"title": "Contribution of Each Enhancement With a Planted Seed",
            "header": ["Configuration", "DSC *M*", "Success", "Leakage", "Mean rank", "*r* vs. full"],
            "rows": rows, "widths": [2.4, 1, 1, 1, 1, 1],
            "note": ("Same planted seeds in every row; each ablated configuration removes exactly one enhancement. Mean "
                     "rank is the average DSC rank among the four ESRG configurations (1 = best) from the Friedman test: "
                     f"χ²({A['df']}, *N* = {A['n']:,}) = {A['chi2']:.2f}, *p* {_pv(A['p'])}, Kendall’s *W* = "
                     f"{nz(A['kendall_w'])} (Section 3.1.2). *r* compares DSC with full ESRG (positive = full ESRG higher); "
                     f"after Holm adjustment, {_p_all(ps)}. The baseline is listed for reference only.")}


def c_time(R):
    T = R["e6"]
    rows = [[CL[lv], f(T[lv]["srg"]["median"], 2), f(T[lv]["esrg"]["median"], 2), pct(T[lv]["reduction_pct_median"]),
             nz(T[lv]["r"])] for lv in LEVELS]
    return {"title": "Processing Time per Slice Under Automatic Seeding",
            "header": ["Tumor class", "SRG *Mdn*, s", "ESRG *Mdn*, s", "Reduction", "*r*"],
            "rows": rows, "widths": [1.8, 1.3, 1.3, 1.2, 0.8],
            "note": (f"*N* = {T['all']['N']} slices ({T['all']['N'] // 9} per stratum) timed in the sequential pass of "
                     "Section 3.1.2. Reduction is 1 − *Mdn*_{ESRG}/*Mdn*_{SRG}; a negative *r* means ESRG was faster "
                     "(Wilcoxon signed-rank test); after Holm adjustment over the three classes, "
                     f"{_p_all([T[lv]['p_holm'] for lv in LEVELS])}. Means and standard deviations are given in Appendix G.")}


def c_stages(R):
    S = R["e6"]["stages"]
    names = [("t_input", "1 · Input"), ("t_mask", "2 · Head mask"), ("t_log", "3 · Log domain"),
             ("t_candidates", "4–5 · Seed selection"), ("t_growth", "6 · Region growing"), ("t_final", "7 · Post-processing")]
    rows = [[n, f(S["A_SRG"][k]["mean"], 3), f(S["A_ESRG"][k]["mean"], 3)] for k, n in names]
    return {"title": "Mean Processing Time per Pipeline Stage",
            "header": ["Stage", "SRG *M*, s", "ESRG *M*, s"],
            "rows": rows, "widths": [2.6, 1.4, 1.4], "align": ["l", "c", "c"],
            "note": ("Same sequential pass as Table [[T:time]]. Stages 1–5 are shared by both algorithms, so their times "
                     "differ only by measurement noise; the difference in total time comes from Stage 6.")}


CH_ORDER = [("overall_auto", lambda R: c_overall(R, "auto")), ("overall_planted", lambda R: c_overall(R, "planted")),
            ("seed_hit", c_seed_hit), ("operator", c_operator), ("buckets", c_buckets), ("recall", c_recall),
            ("bias", c_bias), ("leak", c_leak), ("boundary", c_boundary), ("purify", c_purify),
            ("dsc_class", t_dsc_class), ("ablation", c_ablation), ("time", c_time), ("stages", c_stages)]


def build_chapter(R):
    """Condensed Chapter 4 tables, numbered 4.1, 4.2, … in chapter order."""
    out = {}
    for i, (key, fn) in enumerate(CH_ORDER, 1):
        t = dict(fn(R))
        t["num"] = f"4.{i}"
        out[key] = t
    for t in out.values():
        if t.get("note"):
            t["note"] = re.sub(r"\[\[T:(\w+)\]\]", lambda m: out[m.group(1)]["num"], t["note"])
    return out
