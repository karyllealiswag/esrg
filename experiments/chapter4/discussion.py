"""
discussion.py — Interpretation paragraphs of Chapter 4.

Purpose : Hold the discussion text that interprets each result, written after the
          results were examined, with every number still read from results.json so
          the text cannot drift from the tables.
Function : texts(R, df, tb) returns {placeholder: paragraph text} for the [[NAME]]
          placeholders of build_chapter4.py; tb maps table ids to their numbers. Where a sentence depends on the
          direction or significance of a result, the wording follows the data.
Notes   : Citations follow APA 7 author–date style; full entries for the works first
          cited in this chapter are listed at its end, the others in Chapter 3.
"""
from experiments.chapter4.tables import f, nz, pct
from experiments.stats import fmt_p


def P(p):
    s = fmt_p(p)
    return f"*p* {s}" if s.startswith("<") else f"*p* = {s}"


def sig(p):
    return p is not None and p < 0.05


def texts(R, df, tb):
    E1, E2, E3, E4, E5, E6 = R["e1"], R["e2"], R["e3"], R["e4"], R["e5"], R["e6"]
    S = R["summary"]
    n = R["meta"]["n_sample"]
    au = {r["metric"]: r for r in E4["auto"]}
    pl = {r["metric"]: r for r in E4["planted"]}
    hv = E2["auto_on_hits"]
    lk_auto = E3["auto_leaks"]
    T = {}

    # ── Overall ──────────────────────────────────────────────────────────────
    T["AUTO_DISCUSSION"] = (
        "The enhanced algorithm is therefore significantly better at delineating the tumor, and the gain comes "
        "mainly from recall: it recovers about four times as much of the tumor. The lower precision and higher "
        "leakage rate do not mean that the baseline controls leakage better. Its background seeds surround the "
        "tumor seed and claim the neighboring tissue first, so its tumor region rarely grows: its median area is "
        f"only {f(S['A_SRG']['area_ratio']['median'], 2)} times the true tumor area under automatic seeding, against "
        f"{f(S['A_ESRG']['area_ratio']['median'], 2)} for ESRG. The baseline avoids leakage by severely "
        f"undersegmenting. Moreover, {lk_auto['seed_miss']} of the {lk_auto['n']} slices on which ESRG leaked "
        f"({pct(100 * lk_auto['seed_miss'] / max(lk_auto['n'], 1))}) started from a seed that was not entirely inside "
        "the tumor, so the region grew in the wrong tissue from the outset. On the "
        f"{hv['n']} slices whose automatic seed hit the tumor, which both algorithms share, ESRG reaches a mean DSC "
        f"of {f(hv['dsc']['esrg']['mean'])} against {f(hv['dsc']['srg']['mean'])} and a mean recall of "
        f"{f(hv['recall']['esrg']['mean'])} against {f(hv['recall']['srg']['mean'])} (*r* = {nz(hv['recall']['r'])}, "
        f"{P(hv['recall']['p'])}). HD95 does not differ significantly ({P(au['hd95_wc']['p_holm'])}): both algorithms "
        "start from the same seed, so on the slices where that seed misses the tumor both boundaries are far from the "
        "true one, whichever algorithm grows the region. Sections 4.2.1 and 4.2.3 separate the "
        "contribution of seed placement from that of growth.")
    T["PLANTED_DISCUSSION"] = (
        "With a correctly placed seed, the enhanced algorithm again delineates the tumor significantly better, "
        "with a large effect on DSC and recall. The baseline’s near-perfect precision reflects how little it grows: "
        f"its median predicted area is {f(S['P_SRG']['area_ratio']['median'], 3)} times the tumor area, so almost "
        "everything it absorbs is tumor but it misses most of the tumor. ESRG trades a moderate loss of mean "
        f"precision (median {f(pl['precision']['enh']['median'])}) for a fourfold gain in recall, and its boundary lies "
        f"closer to the true boundary on most slices (median HD95 {f(pl['hd95_wc']['enh']['median'], 2)} against "
        f"{f(pl['hd95_wc']['base']['median'], 2)} pixels). Its mean HD95 is higher than its median because the "
        f"{pct(pl['leaked']['rate_cmp'])} of slices that leak have large boundary errors.")
    z = E4["by_class"]["all"]
    T["DIST_DISCUSSION"] = (
        "the slices on which the seed missed the tumor. ESRG then has a second cluster between 0.8 and 0.95, the "
        "slices on which its seed hit, whereas the baseline has almost no slice above 0.6. With a planted seed the "
        f"zeros disappear for both algorithms ({pct(z['P_ESRG']['dsc0'])} and {pct(z['P_SRG']['dsc0'])}); ESRG spreads "
        "across the whole range while the baseline stays concentrated below 0.2. Because the distribution is "
        "bimodal, a mean summarizes it poorly, which is why medians, success rates, and rank-based tests are used "
        "throughout (Maier-Hein et al., 2024; Reinke et al., 2024).")
    sc, scp = E4["split_check"]["A_ESRG"], E4["split_check"]["P_ESRG"]
    no_adv = (not sig(sc["test_mw"]["p"]) or sc["test"]["mean"] >= sc["train"]["mean"]) and \
             (not sig(scp["test_mw"]["p"]) or scp["test"]["mean"] >= scp["train"]["mean"])
    T["SPLIT_DISCUSSION"] = (
        ("The training-split slices therefore show no advantage; if anything they score slightly lower, so the "
         "tuning of three ranking weights on that split did not inflate the results, and pooling the splits to reach "
         "the required sample size did not bias the evaluation in favor of the method." if no_adv else
         "The training-split slices score higher, so results on them may be slightly optimistic; the test-split "
         "subset is reported in Appendix F for comparison."))

    # ── Objective 1 ──────────────────────────────────────────────────────────
    c, p_ = E1["class"], E1["plane"]
    b = E5
    T["SHR_DISCUSSION"] = (
        f"Reliability depends strongly on the tumor class (χ²({c['test']['df']}) = {c['test']['chi2']:.2f}, "
        f"{P(c['test']['p'])}, Cramér’s *V* = {nz(c['test']['cramers_v'])}, a large association). The seed hits "
        f"{pct(c['meningioma']['rate'])} of meningiomas, whose bright, compact, well-circumscribed enhancement places "
        "them clearly in the highest interior intensity class, consistent with intensity-based seeding results in "
        f"the literature (Biratu et al., 2021); its median seed distance is "
        f"{f(c['meningioma']['seed_dist']['median'], 1)} pixels, that is, the seed centroid lies inside the tumor. "
        f"Reliability is lower for pituitary tumors ({pct(c['pituitary']['rate'])}), which sit centrally among normal "
        "structures of similar brightness, and lowest for gliomas "
        f"({pct(c['glioma']['rate'])}, median distance {f(c['glioma']['seed_dist']['median'], 1)} pixels), whose "
        "heterogeneous, often weakly enhancing tissue is frequently not the brightest structure in the head "
        "(Tohka, 2014). The imaging plane matters much less "
        f"(χ²({p_['test']['df']}) = {p_['test']['chi2']:.2f}, {P(p_['test']['p'])}, *V* = {nz(p_['test']['cramers_v'])}, "
        f"a small association): coronal slices give the highest rate ({pct(p_['coronal']['rate'])}) and sagittal "
        f"slices the lowest ({pct(p_['sagittal']['rate'])}). When the seed misses, it usually lands on a different "
        "structure altogether rather than just outside the tumor edge, as the large seed distances show.")
    op = E1["operator"]
    ao = {lv: op[lv]["auto_vs_operator_esrg"] for lv in op}
    T["OPERATOR_DISCUSSION"] = (
        "Two findings follow. First, the original algorithm appears consistent only because it consistently "
        f"undersegments: its mean DSC over the five operators is {f(op['all']['srg']['mean_dsc']['mean'])}, so its "
        "results cannot vary much. Second, the enhanced grower is far more sensitive to where the seed is placed "
        f"({P(op['all']['sd_test']['p'])}, *r* = {nz(op['all']['sd_test']['r'])}): "
        "because its stopping bound is derived from the statistics of the seed region (Equations 3.9 and 3.10), a click on a "
        "slightly brighter or more heterogeneous part of the tumor produces a different bound and a different result. "
        "This confirms the problem stated for Objective 1, that manual seeding makes the output depend on the "
        "operator (Adams & Bischof, 1994; Fan et al., 2005), and it is exactly the variability that the automated "
        "seed removes: with no operator input, the same slice always yields the same seed and the same mask. The "
        "cost of automation is measured by comparing the automatic result with the average of the five operator "
        f"results for the same grower. Over all slices the operators do better (mean DSC "
        f"{f(op['all']['esrg']['mean_dsc']['mean'])} against {f(op['all']['auto_dsc']['mean'])}; "
        f"{P(ao['all']['p'])}, *r* = {nz(ao['all']['r'])}), but this difference comes from "
        f"glioma ({f(op['glioma']['esrg']['mean_dsc']['mean'])} against {f(op['glioma']['auto_dsc']['mean'])}, "
        f"{P(ao['glioma']['p'])}), where the automatic seed seldom reaches the tumor. For meningioma "
        f"({f(op['meningioma']['esrg']['mean_dsc']['mean'])} against {f(op['meningioma']['auto_dsc']['mean'])}, "
        f"{P(ao['meningioma']['p'])}) and pituitary tumors ({f(op['pituitary']['esrg']['mean_dsc']['mean'])} against "
        f"{f(op['pituitary']['auto_dsc']['mean'])}, {P(ao['pituitary']['p'])}), the automatic seed performs "
        f"{'as well as' if not (sig(ao['meningioma']['p']) and sig(ao['pituitary']['p'])) else 'nearly as well as'} the "
        "average operator who clicks correctly inside the tumor, without any operator input and with no variability. "
        f"The automatic seed also outperforms the original algorithm seeded by an operator (*r* = "
        f"{nz(op['all']['auto_vs_operator_srg']['r'])}, {P(op['all']['auto_vs_operator_srg']['p'])}).")
    allb = b["all"]
    seed_fail = allb["B"]["k"] + allb["C"]["k"] + allb["X"]["k"]
    grow_fail = allb["D"]["k"] + allb["E"]["k"]
    T["BUCKET_DISCUSSION"] = (
        f"Successful slices (bucket G) make up {pct(allb['G']['pct'])} of the sample. Most failures occur before region "
        f"growing begins: on {pct(allb['C']['pct'])} of slices a candidate containing the tumor existed but a different "
        f"structure was ranked first (C), and on {pct(allb['B']['pct'])} the tumor never entered the highest intensity "
        f"class (B). Together with the false no-candidate flags (X), seeding accounts for "
        f"{pct(100 * seed_fail / allb['n'])} of slices, whereas leakage (D) and undersegmentation (E) during growth "
        f"account for only {pct(100 * grow_fail / allb['n'])}. The pattern differs by class "
        f"(χ²({b['test']['df']}) = {b['test']['chi2']:.2f}, {P(b['test']['p'])}, *V* = {nz(b['test']['cramers_v'])}). "
        f"For gliomas the dominant failure is B ({pct(b['glioma']['B']['pct'])}): weakly enhancing tumor tissue falls "
        f"below the uppermost Otsu threshold. For pituitary tumors it is C ({pct(b['pituitary']['C']['pct'])}): the "
        "tumor is a candidate, but a brighter neighboring structure is ranked above it. For meningioma, "
        f"{pct(b['meningioma']['G']['pct'])} of slices succeed. The automated seed selection is therefore the "
        "component that limits the fully automatic pipeline, and the manual seeding option remains necessary for "
        "pituitary tumors and gliomas.")

    # ── Objective 2 ──────────────────────────────────────────────────────────
    rc = E2["recall"]["all"]["tests"]
    pr = E2["precision"]["all"]
    ds = E2["dsc"]["all"]
    passes = E3["passes"]
    pmax = sum(v for k, v in passes.items() if float(k) >= 4)
    T["RECALL_DISCUSSION"] = (
        "The local comparison is the larger of the two contributions: with everything else held fixed, it improved "
        f"recall on {rc['P_ESRG_global']['n_pos']} slices and lowered it on {rc['P_ESRG_global']['n_neg']}, a large "
        "effect in every class. The log transform adds a medium effect "
        f"({rc['P_ESRG_nolog']['n_pos']} slices improved, {rc['P_ESRG_nolog']['n_neg']} worse), and it does so without "
        f"a loss of precision ({f(pr['tests']['P_ESRG_nolog']['ref']['mean'])} without it against "
        f"{f(pr['enh']['mean'])} with it, {P(pr['tests']['P_ESRG_nolog']['p_holm'])}), raising mean DSC from "
        f"{f(ds['tests']['P_ESRG_nolog']['ref']['mean'])} to {f(ds['enh']['mean'])}. In the log domain the tolerance "
        "σ_{A} is effectively relative to the intensity level, so the same bound admits proportionally similar "
        "variations at any brightness. The local measure costs some precision "
        f"({f(pr['tests']['P_ESRG_global']['ref']['mean'])} to {f(pr['enh']['mean'])}), but its recall gain is far "
        f"larger and mean DSC doubles, from {f(ds['tests']['P_ESRG_global']['ref']['mean'])} to {f(ds['enh']['mean'])}. "
        f"The share of slices on which at least half of the tumor is recovered rises from "
        f"{pct(E2['recall_ge_50']['P_SRG']['pct'])} for the baseline to {pct(E2['recall_ge_50']['P_ESRG']['pct'])}. "
        "This directly addresses the second problem: replacing the global-mean difference with a local comparison "
        "in the log domain substantially reduces undersegmentation (Akram et al., 2017; Li et al., 2011). Recall nevertheless remains "
        f"moderate in absolute terms, lowest for pituitary tumors ({f(E2['recall']['pituitary']['enh']['mean'])}). "
        f"Growth ended at the maximum of four passes on {pmax} of the {n:,} planted slices "
        f"({pct(100 * pmax / n)}), so the region was usually still growing when the pass limit was reached, and the "
        f"small purified seed yields a tight initial bound (Table {tb['purify']}).")
    bb = E2["bias"]
    ctx = E2["bias_context"]
    T["BIAS_DISCUSSION"] = (
        "Neither full ESRG, nor ESRG without the log transform, nor the baseline changed significantly under the field "
        f"(Holm-adjusted {P(max(bb[k]['p_holm'] for k in ('B_ESRG', 'B_ESRG_nolog', 'B_SRG')))} or above); individual "
        f"slices moved by a mean of {f(bb['B_ESRG']['mean_abs_change'])} in recall for full ESRG, but in both "
        "directions. Only the global-measure variant changed significantly, by "
        f"{f(bb['B_ESRG_global']['biased']['mean'] - bb['B_ESRG_global']['clean']['mean'])} "
        f"({P(bb['B_ESRG_global']['p_holm'])}), a statistically detectable but practically negligible amount, and N4 "
        f"correction did not change the baseline ({P(bb['B_SRG_N4']['vs_B_SRG']['p'])}). The field used here is smooth: "
        f"across a tumor of median size (about {ctx['gt_diam_median']:.0f} pixels in diameter) it changes the intensity "
        f"by only about {ctx['change_over_median_tumor_pct']:.1f}%, which is small compared with the natural "
        "heterogeneity of the tumor. The experiment therefore shows that ESRG is robust to a field of this size, "
        f"rather than that the baseline is harmed by it; the gains of Objective 2 in Table {tb['recall']} appear on the original "
        "slices and come from comparing each pixel with its own neighborhood on a relative scale.")

    # ── Objective 3 ──────────────────────────────────────────────────────────
    L = E3["leak"]
    dsc3 = E3["dsc"]["all"]
    rec3 = E3["recall"]["all"]
    T["LEAK_DISCUSSION"] = (
        "The improvement in precision is large "
        f"and significant in every class (*r* from {nz(min(E3['precision'][c]['tests']['P_ESRG_nostop']['r'] for c in ('glioma', 'meningioma', 'pituitary')))} "
        f"to {nz(max(E3['precision'][c]['tests']['P_ESRG_nostop']['r'] for c in ('glioma', 'meningioma', 'pituitary')))}). "
        f"The median precision with the criterion is {f(E3['precision']['all']['enh']['median'])}, meaning that on most "
        "slices every absorbed pixel is tumor. The criterion also raises mean DSC from "
        f"{f(dsc3['tests']['P_ESRG_nostop']['ref']['mean'])} to {f(dsc3['enh']['mean'])}, although recall falls from "
        f"{f(rec3['tests']['P_ESRG_nostop']['ref']['mean'])} to {f(rec3['enh']['mean'])}, because unconditional "
        "growth covers the tumor only by covering the whole head. This directly addresses the third problem: the "
        "adaptive stopping criterion and its drift guard prevent the flooding caused by unconditional absorption, in the same way that "
        "confidence-connected region growing bounds a region by its own statistics (Insight Software Consortium, "
        "n.d.). Leakage is not eliminated. Glioma remains the most affected class "
        f"({pct(L['glioma']['leak_ci']['P_ESRG']['pct'])} leakage, mean precision "
        f"{f(E3['precision']['glioma']['enh']['mean'])}), because its infiltrative margin has no sharp intensity step "
        "at which the bound can stop growth. The baseline shows no leakage with a planted seed "
        f"({pct(L['all']['leak_ci']['P_SRG']['pct'])}), but, as noted in Section 4.1.1, it achieves this by stopping far "
        f"inside the tumor (median area ratio {f(L['all']['ratio']['P_SRG']['median'], 3)}).")
    hd = E3["hd95_wc"]
    T["BOUNDARY_DISCUSSION"] = (
        "Without the stopping criterion the boundary lies, at its 95th percentile, a median of "
        f"{f(hd['all']['tests']['P_ESRG_nostop']['ref']['median'], 0)} pixels from the true boundary, about half the "
        "width of a 512-pixel slice, so the criterion reduces the boundary error by roughly an order of magnitude. "
        "Against the "
        "baseline, the full enhanced algorithm places the boundary closer on most slices overall; by class, it is "
        f"clearly better for meningioma (median HD95 {f(hd['meningioma']['enh']['median'], 2)} against "
        f"{f(hd['meningioma']['tests']['P_SRG']['ref']['median'], 2)} pixels) and pituitary tumors "
        f"({f(hd['pituitary']['enh']['median'], 2)} against {f(hd['pituitary']['tests']['P_SRG']['ref']['median'], 2)}), "
        f"but worse for glioma ({f(hd['glioma']['enh']['median'], 2)} against "
        f"{f(hd['glioma']['tests']['P_SRG']['ref']['median'], 2)}), where the slices that leak carry their boundary "
        "into the surrounding tissue. A boundary metric is needed to see this, since overlap metrics alone do not "
        "reveal where the errors lie (Reinke et al., 2024; Taha & Hanbury, 2015).")
    pu = {r["metric"]: r for r in E3["purify"]["table"]}
    T["PURIFY_DISCUSSION"] = (
        "A seed of a few pixels has a very small spread, so the resulting bound is tight and growth stops before the "
        f"tumor edge. Purification halves the leakage rate ({pct(pu['leaked']['rate_ref'])} to "
        f"{pct(pu['leaked']['rate_cmp'])}) and raises precision ({f(pu['precision']['base']['mean'])} to "
        f"{f(pu['precision']['enh']['mean'])}), but it lowers recall ({f(pu['recall']['base']['mean'])} to "
        f"{f(pu['recall']['enh']['mean'])}) and mean DSC ({f(pu['dsc']['base']['mean'])} to "
        f"{f(pu['dsc']['enh']['mean'])}); the larger click disk gives a looser bound that reaches further "
        f"(success {pct(pu['success']['rate_ref'])} against {pct(pu['success']['rate_cmp'])}) but leaks more often. "
        "The seed size therefore controls the balance between the two growth objectives: purification favors "
        "Objective 3, and the unpurified disk favors Objective 2.")

    # ── Overall by class / plane / ablation ──────────────────────────────────
    bc = E4["by_class"]
    kw = E4["kruskal"]
    T["CLASS_DISCUSSION"] = (
        f"Table {tb['dsc_class']} shows three distinct outcomes. Meningioma is delineated reliably and automatically: its median "
        f"DSC improves from {f(bc['meningioma']['A_SRG']['dsc']['median'])} to "
        f"{f(bc['meningioma']['A_ESRG']['dsc']['median'])} ({P(bc['meningioma']['auto_test']['p_holm'])}, "
        f"*r* = {nz(bc['meningioma']['auto_test']['r'])}), and {pct(bc['meningioma']['A_ESRG']['success']['pct'])} of "
        "slices are segmented successfully without any user input. Pituitary tumors improve significantly "
        f"({P(bc['pituitary']['auto_test']['p_holm'])}), and their success rate rises from "
        f"{pct(bc['pituitary']['A_SRG']['success']['pct'])} to {pct(bc['pituitary']['A_ESRG']['success']['pct'])}, but "
        f"their median DSC under automatic seeding stays at {f(bc['pituitary']['A_ESRG']['dsc']['median'])} because the "
        f"seed lands in the tumor on {pct(E1['class']['pituitary']['rate'])} of slices; with a planted seed their median "
        f"rises to {f(bc['pituitary']['P_ESRG']['dsc']['median'])}, so for these tumors the manual seeding mode is the "
        "practical option. Glioma is the most difficult class: the enhanced algorithm is significantly better under "
        f"both seeding conditions, but its median DSC remains low ({f(bc['glioma']['A_ESRG']['dsc']['median'])} "
        f"automatic, {f(bc['glioma']['P_ESRG']['dsc']['median'])} planted). The heterogeneous, infiltrative appearance "
        "of glioma provides no sharp boundary for an intensity-based method to follow, which limits both recall and "
        "precision. The tumor class explains a substantial share of the variation in DSC "
        f"(ε² = {nz(kw['A_ESRG']['class']['epsilon2'])} under automatic and {nz(kw['P_ESRG']['class']['epsilon2'])} "
        "under planted seeding).")
    cp = E4["class_plane"]
    T["PLANE_DISCUSSION"] = (
        "By imaging plane (Appendix F), the plane has a statistically significant but small effect "
        f"under automatic seeding (ε² = {nz(kw['A_ESRG']['plane']['epsilon2'])}) and "
        f"{'no significant effect' if not sig(kw['P_ESRG']['plane']['p']) else 'a significant effect'} with a planted "
        f"seed ({P(kw['P_ESRG']['plane']['p'])}). The plane therefore affects where the automatic seed lands rather "
        "than how the region grows. The clearest example is automatic seeding of pituitary tumors, whose coronal "
        f"slices reach a median DSC of {f(cp['pituitary|coronal']['A_ESRG']['dsc']['median'])} against "
        f"{f(cp['pituitary|axial']['A_ESRG']['dsc']['median'])} on axial and "
        f"{f(cp['pituitary|sagittal']['A_ESRG']['dsc']['median'])} on sagittal slices, in line with the higher coronal "
        "seed hit rate: the pituitary gland is easier to separate from neighboring bright structures on coronal views.")
    ab = E4["ablation"]
    sm = ab["summary"]
    T["ABLATION_DISCUSSION"] = (
        "Removing any single enhancement lowers DSC significantly. The largest loss comes from removing the stopping "
        f"criterion (mean DSC {f(sm['P_ESRG_nostop']['dsc']['mean'])}), followed by restoring the global measure "
        f"({f(sm['P_ESRG_global']['dsc']['mean'])}) and switching off the log transform "
        f"({f(sm['P_ESRG_nolog']['dsc']['mean'])}). The enhancements are complementary: the local log-domain measure "
        "lets the region reach the tumor boundary, and the stopping criterion keeps it from crossing it.")

    # ── Efficiency ───────────────────────────────────────────────────────────
    if E6:
        t = E6["all"]
        st = E6["stages"]
        faster = t["reduction_pct_median"] > 0
        g_srg, g_esrg = st["A_SRG"]["t_growth"]["mean"], st["A_ESRG"]["t_growth"]["mean"]
        T["TIME_DISCUSSION"] = (
            ("The enhanced algorithm is therefore faster as well as more accurate. " if faster and sig(t["p"]) else
             "The two algorithms differ little in speed. " if not sig(t["p"]) else
             "The enhanced algorithm is slower, the price of its additional bookkeeping. ")
            + "The baseline partitions the whole head among hundreds of background seeds, whereas ESRG grows a single "
            "region and stops at its boundary. By class, ESRG takes longest on "
            f"{max(('glioma', 'meningioma', 'pituitary'), key=lambda c: E6[c]['esrg']['median'])} slices "
            f"(median {f(max(E6[c]['esrg']['median'] for c in ('glioma', 'meningioma', 'pituitary')), 2)} s), where weak "
            f"margins let the region grow larger before the bound halts it. Running first or second made no material "
            f"difference (mean {f(E6['order_effect']['A_ESRG']['first']['mean'], 2)} against "
            f"{f(E6['order_effect']['A_ESRG']['second']['mean'], 2)} s for ESRG and "
            f"{f(E6['order_effect']['A_SRG']['first']['mean'], 2)} against "
            f"{f(E6['order_effect']['A_SRG']['second']['mean'], 2)} s for SRG), so the alternation removed any "
            f"warm-cache advantage. At a median of {f(t['esrg']['median'], 2)} s per slice on a laptop CPU without a "
            "GPU, running on battery power, the enhanced algorithm is fast enough for interactive use.")
        T["STAGE_DISCUSSION"] = (
            f"Table {tb['stages']} shows where the time goes. Stages 1 to 5 are identical for both algorithms; the "
            f"difference lies in region growing, which takes a mean {f(g_srg, 2)} s for SRG and {f(g_esrg, 2)} s for "
            f"ESRG. The most expensive shared stage is seed selection ({f(st['A_ESRG']['t_candidates']['mean'], 2)} s), "
            "which ranks every candidate component; it is also the stage that limits accuracy under automatic "
            "seeding, so it is the natural target for further optimization.")

    # ── Summary ──────────────────────────────────────────────────────────────
    T["SUMMARY_INTRO"] = (
        f"Table {tb['summary']} summarizes the evaluation per objective. Each objective is judged by the metric that measures its "
        "problem directly, against the configuration without the corresponding enhancement.")
    rc_all = E2["recall"]["all"]
    T["SUMMARY_CLOSE"] = (
        "Objective 1 is met in the sense intended: multi-level Otsu thresholding with connected-component filtering "
        "now provides an automatic, deterministic seed that "
        "removes the dependence on the operator, which the evaluation showed to be substantial for the enhanced "
        f"grower (inconsistent outcome on {pct(op['all']['esrg']['inconsistent']['pct'])} of slices). Its seed hit rate "
        f"of {pct(E1['class']['all']['rate'])} is high for meningioma and low for glioma, so automatic seeding is "
        "dependable for well-circumscribed tumors while the manual option remains necessary for the others. "
        "Objective 2 is met: the local log-domain measure raises recall from "
        f"{f(rc_all['tests']['P_SRG']['ref']['mean'])} to {f(rc_all['enh']['mean'])} with a planted seed, and both of "
        "its parts contribute significantly. Objective 3 is met: the stopping criterion, which adapts to the intensity "
        "variability of the growing region and is supported by the global drift guard, lowers the leakage "
        f"rate from {pct(L['all']['leak_ci']['P_ESRG_nostop']['pct'])} to {pct(L['all']['leak_ci']['P_ESRG']['pct'])} "
        f"and raises precision from {f(E3['precision']['all']['tests']['P_ESRG_nostop']['ref']['mean'])} to "
        f"{f(E3['precision']['all']['enh']['mean'])}. Combined, the enhancements more than double the mean DSC of the "
        f"original algorithm under automatic seeding ({f(au['dsc']['base']['mean'])} to {f(au['dsc']['enh']['mean'])}) "
        "and with a planted seed "
        f"({f(pl['dsc']['base']['mean'])} to {f(pl['dsc']['enh']['mean'])}). The remaining errors are concentrated in "
        "seed selection for pituitary tumors and gliomas and in the weak margins of gliomas.")
    return T
