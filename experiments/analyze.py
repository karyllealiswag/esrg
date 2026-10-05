"""
analyze.py — Statistics and figures for Chapter 4.

Purpose : Turn the raw per-slice rows written by evaluate.py into every number,
          table, and figure reported in Chapter 4, so each value traces back to a
          per-slice row of the appendix workbooks.
Function : analyze() computes, per evaluation:
            E1  Objective 1 — seed hit rate, seed localization, operator variability,
                determinism of the automatic seed, effect of a hit on delineation;
            E2  Objective 2 — recall (and the missed share) against the baseline and
                the ablated measures; robustness to a synthetic bias field;
            E3  Objective 3 — precision, leakage rate, area ratio, HD95, ASSD
                against the configuration without the stopping criterion;
            E4  overall delineation (DSC, IoU, success) by class and plane, the
                ablation (Friedman), population-weighted estimates;
            E5  failure attribution; E6  sequential processing time.
          figures() draws the Chapter 4 figures; results.json keeps every value.
Notes   : Paired tests are Wilcoxon signed-rank (Z, r = Z/sqrt(N), Holm-adjusted
          within each family); paired rates use the exact McNemar test.
          CLI: python experiments/analyze.py [--dir outputs/evaluation]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.attribution import BUCKET_LABEL
from experiments.evaluate import CONFIGS, INU, N_CLICKS, SUCCESS_DSC
from experiments.stats import (chi2, desc, friedman, holm, kruskal, mannwhitney, mcnemar,
                               shapiro, stratified_mean, wilcoxon, wilson)

CLASSES = ["glioma", "meningioma", "pituitary"]
PLANES = ["axial", "coronal", "sagittal"]
LEVELS = CLASSES + ["all"]
O_SRG = [f"O_SRG_{k}" for k in range(1, N_CLICKS + 1)]
O_ESRG = [f"O_ESRG_{k}" for k in range(1, N_CLICKS + 1)]
BOOL_COLS = ("leaked", "success", "seed_hit", "biased")


# ─── Data ────────────────────────────────────────────────────────────────────
def _bool(s):
    return s.map(lambda v: True if v in (True, "True", 1, "1") else
                 False if v in (False, "False", 0, "0") else None)


def load(d):
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    sample = pd.read_csv(os.path.join(d, "sample.csv"))
    df = df[df.file.isin(set(sample.file))].copy()
    for c in BOOL_COLS:
        df[c] = _bool(df[c])
    return df, sample


def wide(df, cfg, col):
    return df[df.config == cfg].set_index("file")[col]


def pair(df, a, b, col, files=None):
    x, y = wide(df, a, col), wide(df, b, col)
    idx = x.index.intersection(y.index)
    if files is not None:
        idx = idx.intersection(pd.Index(files))
    return x.loc[idx], y.loc[idx]


def files_of(df, level, key="tumor"):
    s = df[df.config == "A_ESRG"]
    return s.file if level == "all" else s[s[key] == level].file


def rate(v):
    v = [bool(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return {"k": sum(v), "n": len(v), "pct": 100.0 * sum(v) / len(v) if v else float("nan"),
            "ci": wilson(sum(v), len(v))}


def compare(df, base, enh, metrics, rates=("success", "leaked"), files=None, worst_case=True):
    """Descriptives of both configurations, Wilcoxon per metric, McNemar per rate; Holm over all."""
    rows, ps = [], []
    for m in metrics:
        x, y = pair(df, base, enh, m, files)
        rows.append({"metric": m, "base": desc(x), "enh": desc(y), **wilcoxon(x, y),
                     "shapiro_diff": shapiro(x, y)})
        ps.append(rows[-1]["p"])
    for m in rates:
        x, y = pair(df, base, enh, m, files)
        x, y = x.fillna(False).astype(bool), y.fillna(False).astype(bool)
        rows.append({"metric": m, **mcnemar(x, y), "ci_ref": wilson(int(x.sum()), len(x)),
                     "ci_cmp": wilson(int(y.sum()), len(y))})
        ps.append(rows[-1]["p"])
    for r, pa in zip(rows, holm(ps)):
        r["p_holm"] = pa
    return rows


def by_class_tests(df, refs, enh, col):
    """Full ESRG (enh) against each reference configuration, per class; Holm across refs within a class."""
    out = {}
    for lv in LEVELS:
        f = files_of(df, lv)
        row = {"enh": desc(wide(df, enh, col).loc[f])}
        tests = {r: wilcoxon(*pair(df, r, enh, col, f)) for r in refs}
        for r, pa in zip(refs, holm([tests[r]["p"] for r in refs])):
            tests[r]["p_holm"] = pa
            tests[r]["ref"] = desc(wide(df, r, col).loc[f])
        row["tests"] = tests
        out[lv] = row
    return out


# ─── E1: Objective 1 ─────────────────────────────────────────────────────────
def e1(df):
    a = df[df.config == "A_ESRG"].copy()
    a["seeded"] = a.seed_area > 0
    R = {}
    for grp, key, levels in (("class", "tumor", CLASSES), ("plane", "plane", PLANES)):
        tab = {}
        for lv in levels + ["all"]:
            s = a if lv == "all" else a[a[key] == lv]
            sd = s[s.seeded]
            k = int((sd.seed_hit == True).sum())
            tab[lv] = {"n": int(len(s)), "no_candidate": int((~s.seeded).sum()),
                       "N_S": int(len(sd)), "hits": k,
                       "rate": 100 * k / len(sd) if len(sd) else float("nan"), "ci": wilson(k, len(sd)),
                       "rate_all": 100 * k / len(s) if len(s) else float("nan"), "ci_all": wilson(k, len(s)),
                       "seed_dist": desc(sd.seed_distance), "inside_frac": desc(sd.seed_in_gt_frac),
                       "seed_area": desc(sd.seed_area)}
        tab["test"] = chi2([[tab[lv]["hits"], tab[lv]["N_S"] - tab[lv]["hits"]] for lv in levels])
        R[grp] = tab
    # class x plane hit rate grid
    R["class_plane"] = {f"{c}|{p}": rate(a[(a.tumor == c) & (a.plane == p) & a.seeded].seed_hit == True)
                        for c in CLASSES for p in PLANES}

    # effect of the seed outcome on delineation
    hit, miss = a[a.seed_hit == True], a[a.seed_hit != True]
    R["hit_vs_miss"] = {"hit": {"dsc": desc(hit.dsc), "success": rate(hit.success)},
                        "miss": {"dsc": desc(miss.dsc), "success": rate(miss.success)},
                        "test": mannwhitney(hit.dsc, miss.dsc)}

    # operator variability: five simulated operator clicks per slice
    ov = {}
    for name, cfgs in (("srg", O_SRG), ("esrg", O_ESRG)):
        m = pd.concat([wide(df, c, "dsc").rename(c) for c in cfgs], axis=1)
        sm = pd.concat([wide(df, c, "success").rename(c) for c in cfgs], axis=1).fillna(False).astype(bool)
        per = pd.DataFrame({"mean": m.mean(axis=1), "sd": m.std(axis=1, ddof=1),
                            "range": m.max(axis=1) - m.min(axis=1),
                            "n_success": sm.sum(axis=1)})
        per["inconsistent"] = (per.n_success > 0) & (per.n_success < len(cfgs))
        ov[name] = {"per": per}
    auto = wide(df, "A_ESRG", "dsc")
    out = {}
    for lv in LEVELS:
        f = files_of(df, lv)
        row = {}
        for name in ("srg", "esrg"):
            p = ov[name]["per"].loc[f]
            row[name] = {"sd": desc(p.sd), "range": desc(p["range"]), "mean_dsc": desc(p["mean"]),
                         "inconsistent": rate(p.inconsistent)}
        row["sd_test"] = wilcoxon(ov["srg"]["per"].loc[f].sd, ov["esrg"]["per"].loc[f].sd)
        row["auto_vs_operator_esrg"] = wilcoxon(ov["esrg"]["per"].loc[f]["mean"], auto.loc[f])
        row["auto_vs_operator_srg"] = wilcoxon(ov["srg"]["per"].loc[f]["mean"], auto.loc[f])
        row["auto_dsc"] = desc(auto.loc[f])
        out[lv] = row
    R["operator"] = out
    R["operator_per"] = {k: v["per"].reset_index().to_dict("records") for k, v in ov.items()}
    # clicks lie inside the tumor by construction: report how deep they were
    oc = df[df.config.isin(O_ESRG)]
    R["operator_clicks"] = {"depth": desc(oc.click_depth), "n": int(len(oc))}
    return R


def determinism(df, timing):
    """Same slice, same configuration, two independent runs (parallel run vs timing pass)."""
    if timing is None or "dsc" not in timing:
        return None
    out = {}
    for cfg in ("A_SRG", "A_ESRG"):
        t = timing[timing.config == cfg].set_index("file")
        r = wide(df, cfg, "dsc")
        ra = wide(df, cfg, "pred_area")
        idx = t.index.intersection(r.index)
        d = (pd.to_numeric(t.loc[idx, "dsc"]) - r.loc[idx]).abs()
        da = (pd.to_numeric(t.loc[idx, "pred_area"]) - ra.loc[idx]).abs()
        same = ((d.fillna(0) == 0) & (da.fillna(0) == 0))
        out[cfg] = {"n": int(len(idx)), "identical": int(same.sum()), "max_abs_dsc_diff": float(d.max()),
                    "pct_identical": 100 * float(same.mean()) if len(idx) else None}
    return out


# ─── E2: Objective 2 ─────────────────────────────────────────────────────────
def e2(df):
    R = {"recall": by_class_tests(df, ["P_SRG", "P_ESRG_global", "P_ESRG_nolog"], "P_ESRG", "recall"),
         "precision": by_class_tests(df, ["P_SRG", "P_ESRG_global", "P_ESRG_nolog"], "P_ESRG", "precision"),
         "dsc": by_class_tests(df, ["P_SRG", "P_ESRG_global", "P_ESRG_nolog"], "P_ESRG", "dsc")}
    for c in ("P_SRG", "P_ESRG_global", "P_ESRG_nolog", "P_ESRG"):
        s = df[df.config == c]
        R.setdefault("missed", {})[c] = desc(1 - s.recall)
        R.setdefault("recall_ge_50", {})[c] = rate(s.recall >= 0.5)
    # automatic seeding, restricted to slices whose seed hit the tumor (same seed for both)
    hits = df[(df.config == "A_ESRG") & (df.seed_hit == True)].file
    R["auto_on_hits"] = {m: {"srg": desc(wide(df, "A_SRG", m).loc[hits]),
                             "esrg": desc(wide(df, "A_ESRG", m).loc[hits]),
                             **wilcoxon(*pair(df, "A_SRG", "A_ESRG", m, hits))}
                         for m in ("recall", "dsc", "precision")}
    R["auto_on_hits"]["n"] = int(len(hits))
    # bias field
    bias, ps = {}, []
    for clean, biased in (("P_SRG", "B_SRG"), ("P_ESRG_global", "B_ESRG_global"),
                          ("P_ESRG_nolog", "B_ESRG_nolog"), ("P_ESRG", "B_ESRG")):
        x, y = pair(df, clean, biased, "recall")
        dx, dy = pair(df, clean, biased, "dsc")
        bias[biased] = {"clean": desc(x), "biased": desc(y), **wilcoxon(x, y),
                        "mean_abs_change": float(np.nanmean(np.abs(y.to_numpy(float) - x.to_numpy(float)))),
                        "dsc_clean": desc(dx), "dsc_biased": desc(dy)}
        ps.append(bias[biased]["p"])
    for k, pa in zip(list(bias), holm(ps)):
        bias[k]["p_holm"] = pa
    bias["B_SRG_N4"] = {"biased": desc(wide(df, "B_SRG_N4", "recall")),
                        "dsc_biased": desc(wide(df, "B_SRG_N4", "dsc")),
                        "vs_B_SRG": wilcoxon(*pair(df, "B_SRG", "B_SRG_N4", "recall")),
                        "B_ESRG_vs": wilcoxon(*pair(df, "B_SRG_N4", "B_ESRG", "recall"))}
    R["bias"] = bias
    gt = wide(df, "P_ESRG", "gt_area")
    diam = 2 * np.sqrt(gt / np.pi)
    R["bias_context"] = {"gt_diam_median": float(diam.median()),
                         "change_over_median_tumor_pct": float(100 * INU * diam.median() / (math.sqrt(2) * 512))}
    return R


# ─── E3: Objective 3 ─────────────────────────────────────────────────────────
def e3(df):
    R = {}
    for col in ("precision", "dsc", "recall", "hd95_wc", "assd_wc"):
        R[col] = by_class_tests(df, ["P_ESRG_nostop", "P_SRG"], "P_ESRG", col)
    leak = {}
    for lv in LEVELS:
        f = files_of(df, lv)
        row = {}
        for ref in ("P_ESRG_nostop", "P_SRG"):
            x, y = pair(df, ref, "P_ESRG", "leaked", f)
            row[ref] = mcnemar(x.fillna(False).astype(bool), y.fillna(False).astype(bool))
        row["ratio"] = {c: desc(wide(df, c, "area_ratio").loc[f]) for c in ("P_ESRG_nostop", "P_SRG", "P_ESRG")}
        row["fp"] = {c: desc(wide(df, c, "fp").loc[f]) for c in ("P_ESRG_nostop", "P_SRG", "P_ESRG")}
        row["leak_ci"] = {c: rate(wide(df, c, "leaked").loc[f]) for c in ("P_ESRG_nostop", "P_SRG", "P_ESRG")}
        leak[lv] = row
    R["leak"] = leak
    st = df[df.config == "P_ESRG"].stop_reason.fillna("").str.replace(r"[0-9.]+", "#", regex=True)
    R["stop_reasons"] = st.value_counts().to_dict()
    R["passes"] = df[df.config == "P_ESRG"].n_passes.value_counts().sort_index().to_dict()
    # seed purification: the bound is set by the seed's spread (Eq. 3.28)
    pur = compare(df, "P_ESRG_nopurify", "P_ESRG", ["dsc", "recall", "precision", "hd95_wc"])
    R["purify"] = {"table": pur,
                   "seed_area": {c: desc(wide(df, c, "seed_area")) for c in ("P_ESRG_nopurify", "P_ESRG")},
                   "sigma_A": {c: desc(wide(df, c, "sigma_A_initial")) for c in ("P_ESRG_nopurify", "P_ESRG")},
                   "floor_binding": {c: 100 * float((df[df.config == c].sigma_A_initial
                                                     <= df[df.config == c].sigma_floor + 1e-9).mean())
                                     for c in ("P_ESRG_nopurify", "P_ESRG", "A_ESRG")}}
    # automatic seeding, seed-hit slices: leakage of SRG vs ESRG
    hits = df[(df.config == "A_ESRG") & (df.seed_hit == True)].file
    x, y = pair(df, "A_SRG", "A_ESRG", "leaked", hits)
    R["auto_on_hits"] = {"leak": mcnemar(x.fillna(False).astype(bool), y.fillna(False).astype(bool)),
                         "precision": wilcoxon(*pair(df, "A_SRG", "A_ESRG", "precision", hits)),
                         "precision_desc": {c: desc(wide(df, c, "precision").loc[hits]) for c in ("A_SRG", "A_ESRG")}}
    a = df[df.config == "A_ESRG"]
    lk = a[a.leaked == True]
    R["auto_leaks"] = {"n": int(len(lk)), "seed_miss": int((lk.seed_hit != True).sum())}
    return R


# ─── E4: overall delineation and ablation ────────────────────────────────────
def e4(df, strata):
    M = ["dsc", "iou", "precision", "recall", "hd95_wc", "assd_wc"]
    R = {"auto": compare(df, "A_SRG", "A_ESRG", M), "planted": compare(df, "P_SRG", "P_ESRG", M)}
    by = {}
    for lv in LEVELS:
        f = files_of(df, lv)
        row = {}
        for c in ("A_SRG", "A_ESRG", "P_SRG", "P_ESRG"):
            s = df[(df.config == c) & df.file.isin(f)]
            row[c] = {"dsc": desc(s.dsc), "iou": desc(s.iou), "success": rate(s.success),
                      "dsc0": 100 * float((s.dsc == 0).mean())}
        row["auto_test"] = wilcoxon(*pair(df, "A_SRG", "A_ESRG", "dsc", f))
        row["planted_test"] = wilcoxon(*pair(df, "P_SRG", "P_ESRG", "dsc", f))
        by[lv] = row
    for key in ("auto_test", "planted_test"):
        for lv, pa in zip(CLASSES, holm([by[c][key]["p"] for c in CLASSES])):
            by[lv][key]["p_holm"] = pa
        by["all"][key]["p_holm"] = by["all"][key]["p"]
    R["by_class"] = by
    grid = {}
    for c in CLASSES + ["all"]:
        for p in PLANES + ["all"]:
            row = {}
            for k in ("A_ESRG", "P_ESRG"):
                s = df[(df.config == k) & ((df.tumor == c) if c != "all" else True)
                       & ((df.plane == p) if p != "all" else True)]
                row[k] = {"dsc": desc(s.dsc), "success": rate(s.success)}
            grid[f"{c}|{p}"] = row
    R["class_plane"] = grid
    R["kruskal"] = {k: {"class": kruskal([df[(df.config == k) & (df.tumor == c)].dsc for c in CLASSES]),
                        "plane": kruskal([df[(df.config == k) & (df.plane == p)].dsc for p in PLANES])}
                    for k in ("A_ESRG", "P_ESRG")}
    # ablation (E4): four ESRG arms from the same planted seed
    arms = ["P_ESRG", "P_ESRG_global", "P_ESRG_nolog", "P_ESRG_nostop"]
    mat = pd.concat([wide(df, k, "dsc").rename(k) for k in arms], axis=1)
    fr = friedman(mat)
    pw = {k: wilcoxon(mat[k], mat["P_ESRG"]) for k in arms[1:]}
    for k, pa in zip(pw, holm([pw[k]["p"] for k in pw])):
        pw[k]["p_holm"] = pa
    fr["pairwise"] = pw
    fr["summary"] = {k: {"dsc": desc(wide(df, k, "dsc")), "recall": desc(wide(df, k, "recall")),
                         "precision": desc(wide(df, k, "precision")), "success": rate(wide(df, k, "success")),
                         "leaked": rate(wide(df, k, "leaked"))} for k in arms + ["P_SRG"]}
    R["ablation"] = fr
    # population-weighted estimates (equal allocation -> reweight to the population)
    N_h = {f"{r.tumor}|{r.plane}": int(r.N_h) for r in strata.itertuples()}
    R["weighted"] = {c: {m: stratified_mean(df[df.config == c], m, "stratum", N_h) for m in ("dsc", "recall", "precision")}
                     for c in ("A_SRG", "A_ESRG", "P_SRG", "P_ESRG")}
    R["unweighted"] = {c: {m: desc(wide(df, c, m)) for m in ("dsc", "recall", "precision")}
                       for c in ("A_SRG", "A_ESRG", "P_SRG", "P_ESRG")}
    # tuning-overlap check: slices drawn from the training split vs the test split
    a = df[df.config == "A_ESRG"]
    R["split_check"] = {c: {"train": desc(df[(df.config == c) & (df.split == "train")].dsc),
                            "test": desc(df[(df.config == c) & (df.split == "test")].dsc),
                            "test_mw": mannwhitney(df[(df.config == c) & (df.split == "train")].dsc,
                                                   df[(df.config == c) & (df.split == "test")].dsc),
                            "seed_hit_train": rate(a[a.split == "train"].seed_hit == True) if c == "A_ESRG" else None,
                            "seed_hit_test": rate(a[a.split == "test"].seed_hit == True) if c == "A_ESRG" else None}
                        for c in ("A_ESRG", "P_ESRG")}
    R["dsc_hist"] = {c: np.histogram(wide(df, c, "dsc").dropna(), bins=np.linspace(0, 1, 21))[0].tolist()
                     for c in ("A_SRG", "A_ESRG", "P_SRG", "P_ESRG")}
    return R


# ─── E5: failure attribution ─────────────────────────────────────────────────
def e5(df):
    a = df[df.config == "A_ESRG"]
    out = {}
    for lv in LEVELS:
        s = a if lv == "all" else a[a.tumor == lv]
        vc = s.bucket.value_counts()
        out[lv] = {b: {"k": int(vc.get(b, 0)), "pct": 100 * float(vc.get(b, 0)) / max(len(s), 1)}
                   for b in BUCKET_LABEL}
        out[lv]["n"] = int(len(s))
    out["test"] = chi2([[out[c][b]["k"] for b in ("G", "B", "C")] + [out[c]["n"] - sum(out[c][b]["k"] for b in ("G", "B", "C"))]
                        for c in CLASSES])
    return out


# ─── E6: processing time ─────────────────────────────────────────────────────
STAGES = ["t_input", "t_mask", "t_log", "t_candidates", "t_growth", "t_final"]


def e6(timing):
    if timing is None:
        return None
    out = {}
    for lv in LEVELS:
        s = timing if lv == "all" else timing[timing.tumor == lv]
        x = s[s.config == "A_SRG"].set_index("file").seconds
        y = s[s.config == "A_ESRG"].set_index("file").seconds
        idx = x.index.intersection(y.index)
        out[lv] = {"srg": desc(x.loc[idx]), "esrg": desc(y.loc[idx]), **wilcoxon(x.loc[idx], y.loc[idx]),
                   "reduction_pct_mean": 100 * (1 - y.loc[idx].mean() / x.loc[idx].mean()),
                   "reduction_pct_median": 100 * (1 - y.loc[idx].median() / x.loc[idx].median()),
                   "faster_share": 100 * float((y.loc[idx] < x.loc[idx]).mean())}
    for lv, pa in zip(CLASSES, holm([out[c]["p"] for c in CLASSES])):
        out[lv]["p_holm"] = pa
    out["all"]["p_holm"] = out["all"]["p"]
    out["stages"] = {c: {st: desc(timing[timing.config == c][st]) for st in STAGES}
                     for c in ("A_SRG", "A_ESRG")}
    out["order_effect"] = {c: {"first": desc(timing[(timing.config == c) & (timing.order == 1)].seconds),
                               "second": desc(timing[(timing.config == c) & (timing.order == 2)].seconds)}
                           for c in ("A_SRG", "A_ESRG")}
    return out


def analyze(d):
    df, sample = load(d)
    strata = pd.read_csv(os.path.join(d, "sample_strata.csv"))
    size = pd.read_csv(os.path.join(d, "sample_size.csv"))
    tp = os.path.join(d, "timing_sequential.csv")
    timing = pd.read_csv(tp) if os.path.isfile(tp) else None
    if timing is not None:
        timing = timing[timing.file.isin(set(sample.file))]
    env_p = os.path.join(d, "timing_environment.json")
    R = {"meta": {"n_sample": int(len(sample)), "N_population": int(strata.N_h.sum()),
                  "n_rows": int(len(df)), "n_configs": int(df.config.nunique()),
                  "n_errors": int((df.status == "ERROR").sum()),
                  "errors": df[df.status == "ERROR"][["file", "config", "error"]].to_dict("records"),
                  "status_counts": df.status.value_counts().to_dict(),
                  "split_counts": sample.split.value_counts().to_dict(),
                  "inu": INU, "n_clicks": N_CLICKS, "success_dsc": SUCCESS_DSC,
                  "environment": json.load(open(env_p)) if os.path.isfile(env_p) else None},
         "strata": strata.to_dict("records"), "sample_size": size.to_dict("records"),
         "configs": [{"id": k, "desc": v[0], "seeding": v[2], "biased": v[3], "overrides": v[1],
                      "n": int((df.config == k).sum()),
                      "no_candidate": int(((df.config == k) & (df.status == "NO TUMOR CANDIDATE")).sum())}
                     for k, v in CONFIGS.items()]}
    R["summary"] = {k: {**{m: desc(df[df.config == k][m]) for m in
                           ("dsc", "iou", "precision", "recall", "hd95", "hd95_wc", "assd_wc", "area_ratio", "seconds")},
                        "success": rate(df[df.config == k].success), "leaked": rate(df[df.config == k].leaked)}
                    for k in CONFIGS}
    R["e1"] = e1(df)
    R["e1"]["determinism"] = determinism(df, timing)
    R["e2"] = e2(df)
    R["e3"] = e3(df)
    R["e4"] = e4(df, strata)
    R["e5"] = e5(df)
    R["e5_labels"] = BUCKET_LABEL
    R["e6"] = e6(timing)
    R["empty_pred"] = {k: int(((df.config == k) & (df.pred_area == 0)).sum()) for k in CONFIGS}
    return df, timing, R


# ─── Figures ─────────────────────────────────────────────────────────────────
C_ESRG, C_SRG, C_A3, C_A4, C_A5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c4458a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLS_LABEL = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary", "all": "All classes"}


def _style(plt):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def _bars(ax, groups, series, values, errs, colors, ylim=(0, 1), fmt="{:.2f}"):
    w = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, (s, col) in enumerate(zip(series, colors)):
        pos = x - 0.4 + w * (i + 0.5)
        e = errs[i] if errs else None
        ax.bar(pos, values[i], w * 0.92, color=col, label=s, yerr=e,
               error_kw={"elinewidth": 0.9, "capsize": 2.5, "ecolor": INK2})
        for j, (p, v) in enumerate(zip(pos, values[i])):
            top = v + ((e[1][j] if isinstance(e, list) and len(e) == 2 and isinstance(e[0], list) else e[j]) if e else 0)
            ax.text(p, top + 0.015 * (ylim[1] - ylim[0]), fmt.format(v), ha="center", va="bottom",
                    fontsize=7, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)


def figures(df, timing, R, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    os.makedirs(outdir, exist_ok=True)
    S = R["summary"]
    groups = [CLS_LABEL[c] for c in LEVELS]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=300)
        plt.close(fig)

    # Figure 4.2 — seed hit rate by class and plane (Wilson CI)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"width_ratios": [4, 4]})
    for ax, grp, lv, lab in ((axes[0], "class", LEVELS, groups),
                             (axes[1], "plane", PLANES + ["all"], ["Axial", "Coronal", "Sagittal", "All planes"])):
        t = R["e1"][grp]
        v = [t[k]["rate"] for k in lv]
        lo = [t[k]["rate"] - t[k]["ci"][0] for k in lv]
        hi = [t[k]["ci"][1] - t[k]["rate"] for k in lv]
        x = np.arange(len(lv))
        ax.bar(x, v, 0.55, color=C_ESRG, yerr=[lo, hi], error_kw={"elinewidth": 0.9, "capsize": 3, "ecolor": INK2})
        for i, val in enumerate(v):
            ax.text(i, val + hi[i] + 1.5, f"{val:.1f}%", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, lab, fontsize=8.5)
        ax.set_ylim(0, 105)
    axes[0].set_title("(a) By tumor class", fontsize=10, loc="left")
    axes[1].set_title("(b) By imaging plane", fontsize=10, loc="left")
    axes[0].set_ylabel("Seed hit rate, % (95% CI)")
    save(fig, "fig4_2_seed_hit_rate.png")

    # Figure 4.3 — operator variability: within-slice SD of DSC over five clicks
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    per = {k: pd.DataFrame(v).set_index("file") for k, v in R["e1"]["operator_per"].items()}
    cls = df[df.config == "A_ESRG"].set_index("file").tumor
    data, pos, cols = [], [], []
    for i, c in enumerate(LEVELS):
        f = cls.index if c == "all" else cls[cls == c].index
        for j, (k, col) in enumerate((("srg", C_SRG), ("esrg", C_ESRG))):
            data.append(per[k].loc[f].sd.dropna().to_numpy())
            pos.append(i * 3 + j)
            cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops={"color": INK})
    for b, col in zip(bp["boxes"], cols):
        b.set_facecolor(col)
        b.set_alpha(0.85)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(LEVELS))], groups)
    ax.set_ylabel("Within-slice SD of DSC")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_SRG, label="SRG, operator click"),
                       Patch(color=C_ESRG, label="ESRG, operator click")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    save(fig, "fig4_3_operator_variability.png")

    # Figure 4.4 — Objective 2 recall (a) and Objective 3 precision (b), planted seed
    def cls_stat(cfg, m):
        v, e = [], []
        for c in LEVELS:
            dd = desc(df[(df.config == cfg) & ((df.tumor == c) if c != "all" else True)][m])
            v.append(dd["mean"])
            e.append(dd["ci95"])
        return v, e
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2))
    c2 = [("P_SRG", "SRG (baseline)", C_SRG), ("P_ESRG_global", "ESRG, global measure", C_A3),
          ("P_ESRG_nolog", "ESRG, log transform off", C_A4), ("P_ESRG", "ESRG (full)", C_ESRG)]
    v, e = zip(*[cls_stat(c, "recall") for c, _, _ in c2])
    _bars(axes[0], groups, [n for _, n, _ in c2], v, e, [c for _, _, c in c2], (0, 1.15))
    axes[0].set_title("(a) Objective 2: recall", fontsize=10, loc="left")
    axes[0].set_ylabel("Mean recall (95% CI)")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    c3 = [("P_ESRG_nostop", "ESRG, no stopping criterion", C_A5), ("P_ESRG", "ESRG (full)", C_ESRG)]
    v, e = zip(*[cls_stat(c, "precision") for c, _, _ in c3])
    _bars(axes[1], groups, [n for _, n, _ in c3], v, e, [c for _, _, c in c3], (0, 1.3))
    axes[1].set_title("(b) Objective 3: precision", fontsize=10, loc="left")
    axes[1].set_ylabel("Mean precision (95% CI)")
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)
    for ax in axes:
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    save(fig, "fig4_4_recall_precision.png")

    # Figure 4.5 — bias robustness
    pairs = [("P_SRG", "B_SRG", "SRG"), ("P_ESRG_global", "B_ESRG_global", "ESRG,\nglobal measure"),
             ("P_ESRG_nolog", "B_ESRG_nolog", "ESRG,\nlog off"), ("P_ESRG", "B_ESRG", "ESRG (full)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    _bars(ax, [n for _, _, n in pairs], ["Without bias field", f"With {int(INU * 100)}% bias field"],
          [[S[a]["recall"]["mean"] for a, _, _ in pairs], [S[b]["recall"]["mean"] for _, b, _ in pairs]],
          [[S[a]["recall"]["ci95"] for a, _, _ in pairs], [S[b]["recall"]["ci95"] for _, b, _ in pairs]],
          ["#9ec5ef", C_ESRG], (0, 0.9))
    ax.set_ylabel("Mean recall (95% CI)")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig4_5_bias_field.png")

    # Figure 4.6 — leakage rate with and without the stopping criterion
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    L = R["e3"]["leak"]
    vals = [[L[c]["leak_ci"][k]["pct"] for c in LEVELS] for k in ("P_ESRG_nostop", "P_ESRG")]
    errs = [[[L[c]["leak_ci"][k]["pct"] - L[c]["leak_ci"][k]["ci"][0] for c in LEVELS],
             [L[c]["leak_ci"][k]["ci"][1] - L[c]["leak_ci"][k]["pct"] for c in LEVELS]] for k in ("P_ESRG_nostop", "P_ESRG")]
    _bars(ax, groups, ["ESRG, no stopping criterion", "ESRG (full)"], vals, errs, [C_A5, C_ESRG], (0, 115), fmt="{:.1f}")
    ax.set_ylabel("Leakage rate, % (95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    save(fig, "fig4_6_leakage_rate.png")

    # Figure 4.1 — DSC distributions
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (b, e, title) in zip(axes, [("A_SRG", "A_ESRG", "(a) Automatic seeding"),
                                        ("P_SRG", "P_ESRG", "(b) Seed planted inside the tumor")]):
        for cfg, col, name in [(b, C_SRG, "SRG (baseline)"), (e, C_ESRG, "ESRG (enhanced)")]:
            v = df[df.config == cfg].dsc.dropna()
            ax.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), histtype="step",
                    linewidth=2, color=col, label=name)
        ax.axvline(SUCCESS_DSC, color=INK2, linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("DSC")
    axes[0].set_ylabel("Share of slices (%)")
    axes[0].legend(fontsize=8)
    save(fig, "fig4_1_dsc_distribution.png")

    # Figure 4.7 — processing time per stage
    if R["e6"]:
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        names = ["Input", "Head mask", "Log domain", "Seed selection", "Region growing", "Post-processing"]
        y = np.arange(2)
        left = np.zeros(2)
        pal = ["#cfd8e3", "#9ec5ef", "#1baf7a", "#eda100", "#eb6834", "#c4458a"]
        for st, nm, col in zip(STAGES, names, pal):
            vals = np.array([R["e6"]["stages"][c][st].get("mean", 0) or 0 for c in ("A_SRG", "A_ESRG")])
            ax.barh(y, vals, left=left, color=col, label=nm, height=0.5)
            left += vals
        ax.set_yticks(y, ["SRG", "ESRG"])
        ax.set_xlabel("Mean time per slice (s)")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
        save(fig, "fig4_7_processing_time.png")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/evaluation")
    ap.add_argument("--no-appendix", action="store_true")
    args = ap.parse_args()
    df, timing, R = analyze(args.dir)
    with open(os.path.join(args.dir, "results.json"), "w") as f:
        json.dump(_clean(R), f, indent=1)
    figures(df, timing, R, os.path.join(args.dir, "figures"))
    if not args.no_appendix:
        from experiments.appendix import write_all
        write_all(df, timing, R, args.dir)
    print("results.json, figures/ and appendix/ written to", args.dir)
