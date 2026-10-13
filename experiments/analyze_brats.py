"""
analyze_brats.py — Statistics and decisions of the BraTS evaluation (PROTOCOL_BRATS.md, v3 design).

Purpose : Turn a BraTS raw_results.csv (experiments/evaluate.py) into the pre-specified results:
          descriptives, the combined three-arm E1, seeding consistency, the method pairs, Wilson
          intervals, and the objective verdicts, per tumor type (glioma, meningioma) and pooled.
Function : e1() A_ESRG vs P_ESRG vs P_SRG (Friedman + Kendall's W, post-hoc Wilcoxon with Holm over
          the three pairs, and the seed hit rate of A_ESRG with its Wilson interval); e2() seeding
          consistency (auto vs the three planted placements, placement SD); family() the paired
          method experiments E3-E6; verdicts() applies the decision rules; writes results.json,
          tests_paired.csv, descriptives.csv and summary.md.
Notes   : Paired differences are oriented so that positive = better (HD95: baseline minus enhanced).
          An effect counts only if Holm p < .05 and |r| >= 0.1. Statistics come from
          experiments/stats.py; the small table helpers (load, wide, pair, _test, _call) are below.
          CLI: python -P experiments/analyze_brats.py <results folder>   (default: dev rehearsal)
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

ALPHA, R_MIN = 0.05, 0.1
LOWER_BETTER = {"hd95_wc", "seconds"}

TYPES = ["glioma", "meningioma"]
SCOPES = TYPES + ["pooled"]
METRICS = ["dsc", "recall", "precision", "hd95_wc"]
LABEL = {"dsc": "DSC", "recall": "Recall", "precision": "Precision", "hd95_wc": "HD95 (px)", "seconds": "Time (s)"}
SHR_TARGET = 70.0
E1_ARMS = ["A_ESRG", "P_ESRG", "P_SRG_solo", "P_SRG"]
E1_METRICS = METRICS + ["seconds"]
PLACEMENTS = ["P_SRG", "P_SRG_2", "P_SRG_3"]

# exp: (baseline, enhanced, tested metrics (primary first), guard, objective, what changes)
PAIRED = {
    "E3":  ("P_SRG", "P_SRG_L", ["recall"], "dsc", "2", "LLDM added to SRG"),
    "E4":  ("P_ESRG_noL", "P_ESRG", ["recall"], "dsc", "2", "LLDM removed from ESRG (no cap, intensity domain)"),
    "E5a": ("P_SRG", "P_SRG_solo_S", ["precision", "hd95_wc"], "dsc", "3", "ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed"),
    "E5b": ("P_SRG_solo", "P_SRG_solo_S", ["precision", "hd95_wc"], None, "3", "ASC added to SRG, tumor seed only (degenerate baseline)"),
    "E6":  ("P_ESRG_C_noS", "P_ESRG_C", ["precision", "hd95_wc"], "dsc", "3", "ASC removed from ESRG (compete mode)"),
}

# Secondary, descriptive comparisons added AFTER the TEST run (PROTOCOL_BRATS §10, 2026-10-13); they use
# rows already collected, never enter verdicts(), and are reported as secondary.
SECONDARY = {
    "S1": ("P_SRG_solo", "P_ESRG", ["dsc", "precision", "recall"], None, "secondary",
           "Equal operator effort: one tumor click for both algorithms"),
    "S2": ("A_SRG", "A_ESRG", ["dsc", "precision", "recall"], None, "secondary",
           "Same automatic tumor seed for both algorithms (SRG with planted competing seeds)"),
}


# ─── table helpers ───────────────────────────────────────────────────────────
def load(d):
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    for c in ("dsc", "recall", "precision", "hd95_wc", "seconds"):
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
    """better / worse / no effect by the decision rule (Holm p < .05 and |r| >= 0.1)."""
    if t.get("p_holm") is None or t["p_holm"] >= ALPHA or abs(t.get("r", 0.0)) < R_MIN:
        return "no effect"
    return "better" if t["r"] > 0 else "worse"


def placement_sd(df, scope):
    """Per slice: SD (n - 1) of DSC over the three planted placements."""
    m = pd.concat([wide(df, c, "dsc", scope).rename(c) for c in PLACEMENTS], axis=1).dropna()
    return m.std(axis=1, ddof=1)


def _f(v, n=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{n}f}"


def _p(p):
    return "—" if p is None else ("< .001" if p < 0.001 else f"{p:.3f}".lstrip("0"))


def family(df, exp, scope):
    base, enh, metrics, guard, _, _ = PAIRED[exp] if exp in PAIRED else SECONDARY[exp]
    tested = metrics + ([guard] if guard else [])
    rows = [_test(df, base, enh, m, scope) for m in tested]
    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = pa
        r["call"] = _call(r)
        r["role"] = "guard" if guard and r["metric"] == guard else ("primary" if r["metric"] == metrics[0] else "secondary")
    return {"baseline": base, "enhanced": enh, "tests": rows}


def e1(df, scope):
    """Overall comparison of four configurations: Friedman + Kendall's W per metric, post-hoc Wilcoxon with Holm
    over the six pairs. Time (seconds per slice) is analysed like HD95 (lower is better)."""
    out = {"arms": E1_ARMS, "metrics": {}}
    pairs = [("A_ESRG", "P_SRG_solo"), ("P_ESRG", "P_SRG_solo"), ("A_ESRG", "P_SRG"), ("P_ESRG", "P_SRG"),
             ("A_ESRG", "P_ESRG"), ("P_SRG", "P_SRG_solo")]
    for m in E1_METRICS:
        mat = pd.concat([wide(df, a, m, scope).rename(a) for a in E1_ARMS], axis=1).dropna()
        fr = friedman(-mat if m in LOWER_BETTER else mat)         # friedman() ranks higher = better
        post = {}
        for a, b in pairs:
            x, y = mat[b], mat[a]                                  # enhanced a vs baseline b
            t = wilcoxon(y, x) if m in LOWER_BETTER else wilcoxon(x, y)
            d = (x - y) if m in LOWER_BETTER else (y - x)
            post[f"{a} vs {b}"] = {**t, "median_paired_diff": float(np.nanmedian(d))}
        for k, pa in zip(post, holm([v["p"] for v in post.values()])):
            post[k]["p_holm"] = pa
            post[k]["call"] = _call(post[k])
        out["metrics"][m] = {"n": int(len(mat)), "friedman": fr, "post": post,
                             "desc": {a: desc(mat[a]) for a in E1_ARMS}}
    a = df[df.config == "A_ESRG"]
    a = a if scope == "pooled" else a[a.tumor == scope]
    k, n = int(a.seed_hit.sum()), int(len(a))
    lo, hi = wilson(k, n)
    out["shr"] = {"hits": k, "n": n, "shr": 100 * k / n if n else float("nan"), "ci": [lo, hi],
                  "no_candidate": int((a.status == "NO TUMOR CANDIDATE").sum())}
    return out


def e2(df, auto, scope):
    conds = [auto] + PLACEMENTS
    mat = pd.concat([wide(df, c, "dsc", scope).rename(c) for c in conds], axis=1).dropna()
    fr = friedman(mat)
    post = {c: wilcoxon(mat[c], mat[auto]) for c in PLACEMENTS}
    for c, pa in zip(PLACEMENTS, holm([post[c]["p"] for c in PLACEMENTS])):
        post[c]["p_holm"] = pa
        post[c]["call"] = _call(post[c])
    sd = placement_sd(df, scope)
    return {"auto": auto, "n": int(len(mat)), "friedman": fr, "posthoc_auto_vs": post,
            "dsc": {c: desc(mat[c]) for c in conds}, "placement_sd": desc(sd),
            "auto_not_lower": all(post[c]["call"] != "worse" for c in PLACEMENTS)}


def determinism(d):
    out = {}
    raw = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    for k in (1, 2):
        p = os.path.join(d, f"raw_results_repeat{k}.csv")
        if os.path.isfile(p):
            rep = pd.read_csv(p, low_memory=False)
            for cfg in ("A_ESRG", "A_SRG"):
                m = raw[raw.config == cfg][["file", "mask_sha1"]].merge(
                    rep[rep.config == cfg][["file", "mask_sha1"]], on="file", suffixes=("", "_rep"))
                out[f"{cfg}_repeat{k}"] = {"n": int(len(m)), "identical": int((m.mask_sha1 == m.mask_sha1_rep).sum())}
    return out


def verdicts(R):
    """Decision rules of PROTOCOL_BRATS §8 (v3 numbering); per tumor type and pooled."""
    V = {}
    for sc in SCOPES:
        t = lambda exp, m: next(x for x in R["paired"][exp][sc]["tests"] if x["metric"] == m)
        e1s = R["e1"][sc]["metrics"]["dsc"]["post"]
        overall = (e1s["A_ESRG vs P_SRG"]["call"] == "better") and (e1s["P_ESRG vs P_SRG"]["call"] == "better")
        o1 = {"a_shr_lb_ge_70": bool(R["e1"][sc]["shr"]["ci"][0] >= SHR_TARGET),
              "b_consistency": bool(R["e2"]["E2a"][sc]["placement_sd"].get("median", 0) > 0),
              "c_auto_not_lower": R["e2"]["E2a"][sc]["auto_not_lower"]}
        o2 = all(t(e, "recall")["call"] == "better" and t(e, "dsc")["call"] != "worse" for e in ("E3", "E4"))
        o3 = all(t(e, "precision")["call"] == "better" and t(e, "hd95_wc")["call"] == "better"
                 and t(e, "dsc")["call"] != "worse" for e in ("E5a", "E6"))
        V[sc] = {"overall": overall, "objective1": o1, "objective2": o2, "objective3": o3}
    return V


def write_summary(R, d):
    L = ["# BraTS evaluation results", "",
         f"Set: {R['set']} | n = {R['n_slices']} patients (one slice each) | arms: {len(R['arms'])}. "
         "Calls: Holm p < .05 and |r| >= 0.1; positive median difference = enhanced better.", ""]
    L += ["## E1: A_ESRG vs P_ESRG vs P_SRG", "",
          "| Scope | Metric | Median A_ESRG / P_ESRG / P_SRG_solo / P_SRG | Friedman p | Kendall W | A_ESRG vs P_SRG | P_ESRG vs P_SRG | A_ESRG vs P_ESRG |",
          "|---|---|---|---|---|---|---|---|"]
    for sc in SCOPES:
        for m in METRICS:
            e = R["e1"][sc]["metrics"][m]; ds = " / ".join(_f(e["desc"][a].get("median")) for a in E1_ARMS)
            cells = []
            for k in ("A_ESRG vs P_SRG", "P_ESRG vs P_SRG", "A_ESRG vs P_ESRG"):
                v = e["post"][k]; cells.append(f"{_f(v['median_paired_diff'])} (p {_p(v['p_holm'])}, r {_f(v['r'], 2)}) {v['call']}")
            L.append(f"| {sc} | {LABEL[m]} | {ds} | {_p(e['friedman']['p'])} | {_f(e['friedman']['kendall_w'], 2)} | " + " | ".join(cells) + " |")
    L += ["", "### Seed hit rate (A_ESRG)", "", "| Scope | Hits / n | SHR | 95% CI | LB >= 70% |", "|---|---|---|---|---|"]
    for sc in SCOPES:
        v = R["e1"][sc]["shr"]
        L.append(f"| {sc} | {v['hits']} / {v['n']} | {v['shr']:.1f}% | {v['ci'][0]:.1f}–{v['ci'][1]:.1f}% | {'met' if v['ci'][0] >= SHR_TARGET else 'not met'} |")
    L += ["", "## E2: seeding consistency (DSC)", "",
          "| Exp | Scope | Median auto / P1 / P2 / P3 | Friedman p | Kendall W | Placement SD median [IQR] | Auto not lower |", "|---|---|---|---|---|---|---|"]
    for exp, by in R["e2"].items():
        for sc in SCOPES:
            v = by[sc]; meds = " / ".join(_f(v["dsc"][c].get("median")) for c in [v["auto"]] + PLACEMENTS); sd = v["placement_sd"]
            L.append(f"| {exp} | {sc} | {meds} | {_p(v['friedman']['p'])} | {_f(v['friedman']['kendall_w'], 2)} | "
                     f"{_f(sd.get('median'))} [{_f(sd.get('q1'))}–{_f(sd.get('q3'))}] | {'yes' if v['auto_not_lower'] else 'no'} |")
    L += ["", "## E3–E6: method experiments", "",
          "| Exp | What changes | Scope | Metric | Median base | Median enh | Median diff | p (Holm) | r | Call |", "|---|---|---|---|---|---|---|---|---|---|"]
    for exp, by in R["paired"].items():
        for sc in SCOPES:
            for t in by[sc]["tests"]:
                L.append(f"| {exp} | {PAIRED[exp][5]} | {sc} | {LABEL[t['metric']]}{' (guard)' if t['role'] == 'guard' else ''} | "
                         f"{_f(t['base'].get('median'))} | {_f(t['enh'].get('median'))} | {_f(t['median_paired_diff'])} | "
                         f"{_p(t['p_holm'])} | {_f(t['r'], 2)} | {t['call']} |")
    L += ["", "## Secondary comparisons (added after the TEST run; not part of the verdicts)", "",
          "| Exp | What changes | Scope | Metric | Median base | Median enh | Median diff | p (Holm) | r | Call |", "|---|---|---|---|---|---|---|---|---|---|"]
    for exp, by in R.get("secondary", {}).items():
        for sc in SCOPES:
            for t in by[sc]["tests"]:
                L.append(f"| {exp} | {SECONDARY[exp][5]} | {sc} | {LABEL[t['metric']]} | {_f(t['base'].get('median'))} | "
                         f"{_f(t['enh'].get('median'))} | {_f(t['median_paired_diff'])} | {_p(t['p_holm'])} | {_f(t['r'], 2)} | {t['call']} |")
    L += ["", "## Objective verdicts", "", "| Scope | Overall | Obj 1 SHR LB>=70% | Obj 1 consistency | Obj 1 auto not lower | Obj 2 | Obj 3 |", "|---|---|---|---|---|---|---|"]
    yn = lambda b: "met" if b else "not met"
    for sc, v in R["verdicts"].items():
        o = v["objective1"]
        L.append(f"| {sc} | {yn(v['overall'])} | {yn(o['a_shr_lb_ge_70'])} | {yn(o['b_consistency'])} | {yn(o['c_auto_not_lower'])} | {yn(v['objective2'])} | {yn(v['objective3'])} |")
    L += ["", "## Mean DSC / recall / precision by arm (pooled)", "", "| Arm | DSC | Recall | Precision |", "|---|---|---|---|"]
    for a in R["arms"]:
        s = R["descriptives"][a]["pooled"]
        L.append(f"| {a} | {_f(s['dsc'].get('mean'))} | {_f(s['recall'].get('mean'))} | {_f(s['precision'].get('mean'))} |")
    L += ["", "## Determinism", ""] + [f"- {k}: {v['identical']}/{v['n']} masks identical" for k, v in R["determinism"].items()]
    open(os.path.join(d, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def analyze(d):
    df = load(d)
    arms = sorted(df.config.unique())
    R = {"set": os.path.basename(d.rstrip("/\\")), "n_slices": int(df.file.nunique()), "arms": arms,
         "errors": int(df.error.notna().sum()) if "error" in df else 0,
         "descriptives": {a: {sc: {m: desc(wide(df, a, m, sc)) for m in METRICS} for sc in SCOPES} for a in arms},
         "e1": {sc: e1(df, sc) for sc in SCOPES},
         "e2": {"E2a": {sc: e2(df, "A_ESRG", sc) for sc in SCOPES}, "E2b": {sc: e2(df, "A_SRG", sc) for sc in SCOPES}},
         "paired": {exp: {sc: family(df, exp, sc) for sc in SCOPES} for exp in PAIRED},
         "secondary": {exp: {sc: family(df, exp, sc) for sc in SCOPES} for exp in SECONDARY},
         "determinism": determinism(d)}
    R["verdicts"] = verdicts(R)
    json.dump(R, open(os.path.join(d, "results.json"), "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = []
    for exp, by in R["paired"].items():
        for sc, fam in by.items():
            for t in fam["tests"]:
                rows.append({"experiment": exp, "scope": sc, "baseline": fam["baseline"], "enhanced": fam["enhanced"], "metric": t["metric"],
                             "role": t["role"], "N": t["N"], "base_median": t["base"].get("median"), "enh_median": t["enh"].get("median"),
                             "base_mean": t["base"].get("mean"), "enh_mean": t["enh"].get("mean"), "median_paired_diff": t["median_paired_diff"],
                             "n_better": t["n_pos"], "n_worse": t["n_neg"], "Z": t["Z"], "p": t["p"], "p_holm": t["p_holm"], "r": t["r"], "call": t["call"]})
    pd.DataFrame(rows).to_csv(os.path.join(d, "tests_paired.csv"), index=False)
    pd.DataFrame([{"arm": a, "scope": sc, "metric": m, **{k: R["descriptives"][a][sc][m].get(k) for k in ("n", "median", "q1", "q3", "mean", "sd")}}
                  for a in arms for sc in SCOPES for m in METRICS]).to_csv(os.path.join(d, "descriptives.csv"), index=False)
    write_summary(R, d)
    return R


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/rehearsal_brats/dev"
    analyze(d)
    print(open(os.path.join(d, "summary.md"), encoding="utf-8").read())
