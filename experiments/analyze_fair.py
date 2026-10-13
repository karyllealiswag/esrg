"""
analyze_fair.py — Statistics and decisions of the fair BRISC evaluation (PROTOCOL_BRISC_FAIR.md).

Purpose : Turn outputs/evaluation_fair/raw_results.csv into the pre-specified results of
          Experiments E1–E8: descriptives, paired tests, Holm-adjusted families, Wilson
          intervals, Friedman + Kendall's W for E4, and the objective verdicts.
Function : test_family() runs one experiment family (one scope: a tumor type or pooled);
          e3()/e4() handle seeding; verdicts() applies the decision rules of §6;
          sensitivity() reports plane and BRISC-split tables; determinism() checks the
          repeat runs; old_vs_fair() sets the corrected arms next to the original ones on
          the same slices. Writes results_fair.json, CSV tables and summary.md.
Notes   : Paired differences are oriented so that positive = better (HD95: baseline minus
          enhanced). An effect counts only if Holm p < .05 and r >= 0.1 (§6). HD95 uses
          the worst-case value (image diagonal) for an empty prediction.
          CLI: python experiments/analyze_fair.py
"""
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.stats import desc, friedman, holm, shapiro, wilcoxon, wilson

OUT = "outputs/evaluation_fair"
OLD = "outputs/evaluation/raw_results.csv"
TYPES = ["glioma", "meningioma", "pituitary"]
SCOPES = TYPES + ["pooled"]
ALPHA, R_MIN, SHR_TARGET = 0.05, 0.1, 70.0
LOWER_BETTER = {"hd95_wc"}
LABEL = {"dsc": "DSC", "recall": "Recall", "precision": "Precision", "hd95_wc": "HD95 (px)"}

# exp: (baseline arm, enhanced arm, tested metrics (primary first), guard metric or None, objective)
PAIRED = {
    "E1":  ("P_SRG", "A_ESRG", ["dsc", "recall", "precision", "hd95_wc"], None, "overall"),
    "E2":  ("P_SRG", "P_ESRG", ["dsc", "recall", "precision", "hd95_wc"], None, "overall"),
    "E5":  ("P_SRG", "P_SRG_L", ["recall"], "dsc", "2"),
    "E6":  ("P_ESRG_noL", "P_ESRG", ["recall"], "dsc", "2"),
    "E7a": ("P_SRG", "P_SRG_S", ["precision", "hd95_wc"], "dsc", "3"),
    "E7b": ("P_SRG_solo", "P_SRG_solo_S", ["precision", "hd95_wc"], None, "3"),
    "E8":  ("P_ESRG_C_noS", "P_ESRG_C", ["precision", "hd95_wc"], "dsc", "3"),
}
E4 = {"E4a": "A_ESRG", "E4b": "A_SRG"}
PLACEMENTS = ["P_SRG", "P_SRG_2", "P_SRG_3"]


def load(d=OUT):
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    for c in ("dsc", "recall", "precision", "hd95_wc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["seed_hit"] = df["seed_hit"].astype(str).eq("True")
    return df


def wide(df, cfg, col, scope="pooled"):
    s = df[df.config == cfg]
    if scope != "pooled":
        s = s[s.tumor == scope]
    return s.set_index("file")[col]


def pair(df, base, enh, col, scope):
    x, y = wide(df, base, col, scope), wide(df, enh, col, scope)
    idx = x.index.intersection(y.index)
    return x.loc[idx], y.loc[idx]


def _test(df, base, enh, metric, scope):
    x, y = pair(df, base, enh, metric, scope)
    # Orient so positive = better: wilcoxon(a, b) uses d = b - a.
    t = wilcoxon(y, x) if metric in LOWER_BETTER else wilcoxon(x, y)
    d = (x - y) if metric in LOWER_BETTER else (y - x)
    return {"metric": metric, "base": desc(x), "enh": desc(y),
            "median_paired_diff": float(np.nanmedian(d)) if len(d) else None,
            "shapiro_diff": shapiro(x, y), **t}


def _call(t):
    """better / worse / no effect by the §6 rule (Holm p < .05 and |r| >= 0.1)."""
    if t.get("p_holm") is None or t["p_holm"] >= ALPHA or abs(t.get("r", 0.0)) < R_MIN:
        return "no effect"
    return "better" if t["r"] > 0 else "worse"


def test_family(df, exp, scope):
    base, enh, metrics, guard, _ = PAIRED[exp]
    tested = metrics + ([guard] if guard else [])
    rows = [_test(df, base, enh, m, scope) for m in tested]
    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = pa
        r["call"] = _call(r)
        r["role"] = "guard" if guard and r["metric"] == guard else (
            "primary" if r["metric"] == metrics[0] else "secondary")
    out = {"baseline": base, "enhanced": enh, "tests": rows}
    if exp == "E7b":                                     # DSC descriptive only (degenerate baseline)
        out["dsc_descriptive"] = {"base": desc(wide(df, base, "dsc", scope)), "enh": desc(wide(df, enh, "dsc", scope))}
    return out


def e3(df):
    a = df[df.config == "A_ESRG"]
    out = {}
    for scope in SCOPES:
        s = a if scope == "pooled" else a[a.tumor == scope]
        k, n = int(s.seed_hit.sum()), int(len(s))
        lo, hi = wilson(k, n)
        out[scope] = {"hits": k, "n": n, "shr": 100 * k / n if n else float("nan"), "ci": [lo, hi],
                      "no_candidate": int((s.status == "NO TUMOR CANDIDATE").sum()),
                      "meets_target": (bool(lo >= SHR_TARGET) if scope != "pooled" else None)}
    return out


def placement_sd(df, scope):
    """Per slice: SD (n - 1) of DSC over the three planted placements."""
    m = pd.concat([wide(df, c, "dsc", scope).rename(c) for c in PLACEMENTS], axis=1).dropna()
    return m.std(axis=1, ddof=1)


def e4(df, exp, auto, scope):
    conds = [auto] + PLACEMENTS
    mat = pd.concat([wide(df, c, "dsc", scope).rename(c) for c in conds], axis=1).dropna()
    fr = friedman(mat)
    post = {c: wilcoxon(mat[c], mat[auto]) for c in PLACEMENTS}       # d = auto - placement
    for c, pa in zip(PLACEMENTS, holm([post[c]["p"] for c in PLACEMENTS])):
        post[c]["p_holm"] = pa
        post[c]["call"] = _call(post[c])
    sd = placement_sd(df, scope)
    return {"auto": auto, "n": int(len(mat)), "friedman": fr, "posthoc_auto_vs": post,
            "dsc": {c: desc(mat[c]) for c in conds},
            "placement_sd": desc(sd), "share_sd_gt_0": float((sd > 0).mean()) if len(sd) else None,
            "auto_not_lower": all(post[c]["call"] != "worse" for c in PLACEMENTS)}


def determinism(d=OUT):
    raw = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    out = {}
    for k in (1, 2):
        p = os.path.join(d, f"raw_results_repeat{k}.csv")
        if not os.path.isfile(p):
            continue
        rep = pd.read_csv(p, low_memory=False)
        for cfg in ("A_ESRG", "A_SRG"):
            m = raw[raw.config == cfg][["file", "mask_sha1"]].merge(
                rep[rep.config == cfg][["file", "mask_sha1"]], on="file", suffixes=("", "_rep"))
            out[f"{cfg}_repeat{k}"] = {"n": int(len(m)), "identical": int((m.mask_sha1 == m.mask_sha1_rep).sum())}
    return out


def sensitivity(df):
    """Descriptive: median DSC per arm by plane and by BRISC split (train = tuning split)."""
    arms = sorted(df.config.unique())
    return {"plane": {a: {p: float(df[(df.config == a) & (df.plane == p)].dsc.median())
                          for p in ("axial", "coronal", "sagittal")} for a in arms},
            "split": {a: {s: {"median_dsc": float(df[(df.config == a) & (df.split == s)].dsc.median()),
                              "n": int(((df.config == a) & (df.split == s)).sum())}
                          for s in ("train", "test")} for a in arms},
            "split_paired_median_diff": {
                exp: {s: float(np.nanmedian(
                    wide(df[df.split == s], PAIRED[exp][1], "dsc") - wide(df[df.split == s], PAIRED[exp][0], "dsc")))
                    for s in ("train", "test")} for exp in ("E1", "E2")}}


def old_vs_fair(df):
    """Mean DSC of the original arms (old design) and the corrected arms, on the same slices."""
    if not os.path.isfile(OLD):
        return None
    old = pd.read_csv(OLD, low_memory=False)
    files = set(df.file)
    old = old[old.file.isin(files)]
    rows = {}
    for cfg in ("P_SRG", "P_ESRG", "A_SRG", "A_ESRG"):
        o, n = old[old.config == cfg], df[df.config == cfg]
        rows[cfg] = {"old_mean_dsc": float(o.dsc.mean()), "fair_mean_dsc": float(n.dsc.mean()),
                     "old_median_dsc": float(o.dsc.median()), "fair_median_dsc": float(n.dsc.median()),
                     "n_old": int(len(o)), "n_fair": int(len(n))}
    return rows


def verdicts(R):
    """§6 decision rules, per tumor type and pooled."""
    V = {}
    for scope in SCOPES:
        t = lambda exp, m: next(x for x in R["paired"][exp][scope]["tests"] if x["metric"] == m)
        overall = all(t(e, "dsc")["call"] == "better" for e in ("E1", "E2"))
        o2 = all(t(e, "recall")["call"] == "better" and t(e, "dsc")["call"] != "worse" for e in ("E5", "E6"))
        o3 = all(t(e, "precision")["call"] == "better" and t(e, "hd95_wc")["call"] == "better"
                 and t(e, "dsc")["call"] != "worse" for e in ("E7a", "E8"))
        e4a = R["e4"]["E4a"][scope]
        o1 = {"a_shr": R["e3"][scope]["meets_target"],
              "b_consistency": bool(e4a["placement_sd"].get("median", 0) > 0),
              "c_auto_not_lower": e4a["auto_not_lower"]}
        V[scope] = {"overall": overall, "objective1": o1, "objective2": o2, "objective3": o3}
    return V


def _f(v, n=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{n}f}"


def _p(p):
    return "—" if p is None else ("< .001" if p < 0.001 else f"{p:.3f}".lstrip("0"))


def write_tables(R, d=OUT):
    rows = []
    for exp, by in R["paired"].items():
        for scope, fam in by.items():
            for t in fam["tests"]:
                rows.append({"experiment": exp, "scope": scope, "baseline": fam["baseline"], "enhanced": fam["enhanced"],
                             "metric": t["metric"], "role": t["role"], "N": t["N"],
                             "base_median": t["base"].get("median"), "enh_median": t["enh"].get("median"),
                             "base_mean": t["base"].get("mean"), "enh_mean": t["enh"].get("mean"),
                             "median_paired_diff": t["median_paired_diff"], "n_better": t["n_pos"],
                             "n_worse": t["n_neg"], "n_tied": t["n_zero"], "W_plus": t.get("W_plus"),
                             "Z": t["Z"], "p": t["p"], "p_holm": t["p_holm"], "r": t["r"],
                             "shapiro_p": t["shapiro_diff"].get("p"), "call": t["call"]})
    pd.DataFrame(rows).to_csv(os.path.join(d, "tests_paired.csv"), index=False)
    desc_rows = []
    for arm in R["arms"]:
        for scope in SCOPES:
            for m in ("dsc", "recall", "precision", "hd95_wc"):
                s = R["descriptives"][arm][scope][m]
                desc_rows.append({"arm": arm, "scope": scope, "metric": m, **{k: s.get(k) for k in
                                  ("n", "median", "q1", "q3", "mean", "sd")}})
    pd.DataFrame(desc_rows).to_csv(os.path.join(d, "descriptives.csv"), index=False)


def write_summary(R, d=OUT):
    L = ["# Fair BRISC evaluation (25% subsample): results", "",
         f"n = {R['n_slices']} slices (27 per tumor x plane stratum). Protocol: experiments/PROTOCOL_BRISC_FAIR.md.",
         "Calls use the pre-specified rule: Holm p < .05 and |r| >= 0.1. Positive median difference = enhanced better.", ""]
    L += ["## Effect of the corrections (same slices, mean DSC)", "",
          "| Arm | Original design | Fair design |", "|---|---|---|"]
    for a, v in (R["old_vs_fair"] or {}).items():
        L.append(f"| {a} | {_f(v['old_mean_dsc'])} | {_f(v['fair_mean_dsc'])} |")
    L += ["", "## Paired experiments", "",
          "| Exp | Scope | Baseline → Enhanced | Metric | Median base | Median enh | Median diff | p (Holm) | r | Call |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for exp, by in R["paired"].items():
        for scope in SCOPES:
            fam = by[scope]
            for t in fam["tests"]:
                L.append(f"| {exp} | {scope} | {fam['baseline']} → {fam['enhanced']} | {LABEL[t['metric']]}"
                         f"{' (guard)' if t['role'] == 'guard' else ''} | {_f(t['base'].get('median'))} | "
                         f"{_f(t['enh'].get('median'))} | {_f(t['median_paired_diff'])} | {_p(t['p_holm'])} | "
                         f"{_f(t['r'], 2)} | {t['call']} |")
    L += ["", "## E3 seed hit rate (A_ESRG)", "", "| Scope | Hits / n | SHR | 95% CI | Target LB ≥ 70% |", "|---|---|---|---|---|"]
    for scope, v in R["e3"].items():
        L.append(f"| {scope} | {v['hits']} / {v['n']} | {v['shr']:.1f}% | {v['ci'][0]:.1f}–{v['ci'][1]:.1f}% | "
                 f"{'—' if v['meets_target'] is None else ('met' if v['meets_target'] else 'not met')} |")
    L += ["", "## E4 consistency and accuracy", "",
          "| Exp | Scope | Median DSC auto / P1 / P2 / P3 | Friedman p | Kendall W | Placement SD median [IQR] | Auto not lower |",
          "|---|---|---|---|---|---|---|"]
    for exp, by in R["e4"].items():
        for scope in SCOPES:
            v = by[scope]
            meds = " / ".join(_f(v["dsc"][c].get("median")) for c in [v["auto"]] + PLACEMENTS)
            sd = v["placement_sd"]
            L.append(f"| {exp} | {scope} | {meds} | {_p(v['friedman']['p'])} | {_f(v['friedman']['kendall_w'], 2)} | "
                     f"{_f(sd.get('median'))} [{_f(sd.get('q1'))}–{_f(sd.get('q3'))}] | {'yes' if v['auto_not_lower'] else 'no'} |")
    L += ["", "## Objective verdicts (§6)", "", "| Scope | Overall | Obj 1 (a) SHR | Obj 1 (b) consistency | Obj 1 (c) auto not lower | Obj 2 | Obj 3 |",
          "|---|---|---|---|---|---|---|"]
    yn = lambda b: "—" if b is None else ("met" if b else "not met")
    for scope, v in R["verdicts"].items():
        o1 = v["objective1"]
        L.append(f"| {scope} | {yn(v['overall'])} | {yn(o1['a_shr'])} | {yn(o1['b_consistency'])} | "
                 f"{yn(o1['c_auto_not_lower'])} | {yn(v['objective2'])} | {yn(v['objective3'])} |")
    L += ["", "## Determinism", ""]
    for k, v in R["determinism"].items():
        L.append(f"- {k}: {v['identical']}/{v['n']} masks identical")
    L += ["", "## Sensitivity: tuning split (train) vs held-out split (test), median paired DSC difference", ""]
    for exp, v in R["sensitivity"]["split_paired_median_diff"].items():
        L.append(f"- {exp}: train {_f(v['train'])}, test {_f(v['test'])}")
    with open(os.path.join(d, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def analyze(d=OUT):
    df = load(d)
    errors = int(df.error.notna().sum()) if "error" in df else 0
    arms = sorted(df.config.unique())
    R = {"n_slices": int(df.file.nunique()), "n_rows": int(len(df)), "errors": errors, "arms": arms,
         "descriptives": {a: {s: {m: desc(wide(df, a, m, s)) for m in ("dsc", "recall", "precision", "hd95_wc")}
                              for s in SCOPES} for a in arms},
         "paired": {exp: {s: test_family(df, exp, s) for s in SCOPES} for exp in PAIRED},
         "e3": e3(df),
         "e4": {exp: {s: e4(df, exp, auto, s) for s in SCOPES} for exp, auto in E4.items()},
         "determinism": determinism(d), "sensitivity": sensitivity(df), "old_vs_fair": old_vs_fair(df)}
    R["verdicts"] = verdicts(R)
    with open(os.path.join(d, "results_fair.json"), "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_tables(R, d)
    write_summary(R, d)
    return R


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    R = analyze()
    print(open(os.path.join(OUT, "summary.md"), encoding="utf-8").read())
