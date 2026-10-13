"""
improve_esrg.py — The pre-declared ESRG improvement search (experiments/IMPROVEMENT_PROTOCOL_ESRG.md).

Purpose : Test change 1 (sigma_cap, re-tuned noise floor) and change 2 (flood guard) on the BraTS
          DEVELOPMENT set only, with the same search applied to SRG + ASC, and decide each round by
          the protocol's fixed rule. TEST is never read.
Function : make_folds() splits DEV into folds A and B (stratified, seed 2026); size_prior() gets
          the tumor-size quantiles from fold-A ground truth; screen() runs one round's grid on
          fold A; confirm() runs the winner and the reference on fold B; decide() applies the
          acceptance rule (gain >= 0.02 on B, no tumor type down by more than 0.01); summary()
          compares the final arms with plain SRG on all of DEV (the stop rule) and writes
          configs/esrg_brats_frozen_v2.json.
Notes   : Resumable and crash-safe (rows are appended and flushed; rerunning the same command
          continues). Planted P1 seeds from experiments/evaluate.py. The earlier frozen file
          (configs/esrg_brats_frozen.json) is the starting point and is never overwritten.
          CLI: python -P experiments/improve_esrg.py
"""
import csv
import itertools
import json
import os
import random
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg import preprocessing as pre
from esrg.io_utils import load_image, load_mask
from experiments.evaluate import seed_protocol
from experiments.sampling import SEED, read_split
from experiments.tune import _done, _drive, _repair

OUT = "outputs/improve_esrg"
FROZEN = "configs/esrg_brats_frozen.json"
FROZEN_V2 = "configs/esrg_brats_frozen_v2.json"
TYPES = ("glioma", "meningioma")
GAIN_MIN, TYPE_DROP_MAX, TIE = 0.02, 0.01, 0.005
STOP_GAP = 0.05
FIELDS = ["config", "arm", "fold", "file", "tumor", "dsc", "recall", "precision", "area_ratio",
          "retries", "seconds", "error"]

# Current (already frozen) settings of the two arms.
CUR = {"ESRG": dict(sigma_cap=None, k_local=2.0, k_global=3.0, sigma_floor_min=0.02, flood_frac=None),
       "SRG+S": dict(sigma_cap=None, k_local=5.0, k_global=7.0, sigma_floor_min=0.02, flood_frac=None)}
R1 = {"ESRG": dict(sigma_cap=(None, 0.04, 0.06, 0.08, 0.12), kpair=((2.0, 3.0), (2.5, 3.5), (3.0, 4.0)),
                   sigma_floor_min=(0.02, 0.03)),
      "SRG+S": dict(sigma_cap=(None, 0.06, 0.12), kpair=((3.0, 4.0), (5.0, 7.0), (6.0, 8.0)),
                    sigma_floor_min=(0.02, 0.03))}
R2K = {"ESRG": ((2.0, 3.0), (2.5, 3.5), (3.0, 4.0), (4.0, 5.0)),
       "SRG+S": ((3.0, 4.0), (5.0, 7.0), (6.0, 8.0))}


def cid(arm, p):
    f = lambda v: "none" if v is None else f"{v:g}"
    return (f"{arm}|cap={f(p['sigma_cap'])}|kl={p['k_local']:g}|kg={p['k_global']:g}"
            f"|floor={p['sigma_floor_min']:g}|flood={f(p['flood_frac'])}")


def to_config(arm, p, base):
    """The Config for one arm and parameter set (planted P1 seeds, manual mode)."""
    common = dict(seed_mode="manual", sigma_cap=p["sigma_cap"], sigma_floor_min=p["sigma_floor_min"],
                  flood_frac=p["flood_frac"])
    if arm == "ESRG":
        return base.replace(k_local=p["k_local"], k_global=p["k_global"], **common)
    return base.replace(method="srg", srg_use_stopping=True, srg_k_local=p["k_local"],
                        srg_k_global=p["k_global"], **common)


# ─── Folds and size prior ─────────────────────────────────────────────────────
def make_folds():
    path = os.path.join(OUT, "folds.csv")
    if os.path.isfile(path):
        return dict(csv.reader(open(path)))
    rng = random.Random(SEED)
    fold = {}
    for t in TYPES:
        files = sorted(r["file"] for _, _, r in read_split("brats_eval", "dev") if r["tumor"] == t)
        rng.shuffle(files)
        fold.update({f: "AB"[i % 2] for i, f in enumerate(files)})
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(sorted(fold.items()))
    return fold


def size_prior(folds):
    """P95 and P99 of tumor-core area / brain-mask area over fold-A ground truth (no method output)."""
    path = os.path.join(OUT, "size_prior.json")
    if os.path.isfile(path):
        return json.load(open(path))
    fr = []
    for ip, mp, r in read_split("brats_eval", "dev"):
        if folds[r["file"]] != "A":
            continue
        raw, _ = load_image(ip, 512)
        brain, _, _ = pre.brain_mask(pre.normalize(raw))
        fr.append(float(load_mask(mp, raw.shape).sum() / brain.sum()))
    out = {"n": len(fr), "p95": float(np.percentile(fr, 95)), "p99": float(np.percentile(fr, 99)),
           "max": float(max(fr)), "median": float(np.median(fr))}
    json.dump(out, open(path, "w"), indent=2)
    return out


# ─── Worker ───────────────────────────────────────────────────────────────────
def worker(task):
    ip, mp, info, jobs = task
    base = Config.load(FROZEN)
    raw, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, raw.shape)
    seeds, _ = seed_protocol(raw, gt, base, info["file"])
    rows = []
    for c, arm, p, fold in jobs:
        r = {"config": c, "arm": arm, "fold": fold, "file": info["file"], "tumor": info["tumor"]}
        t0 = time.perf_counter()
        try:
            cfg = to_config(arm, p, base) if arm != "SRG" else base.replace(method="srg", seed_mode="manual")
            res = run(ip, cfg, mask_path=mp, manual_points=seeds[1])
            sc = res.scores
            r.update(dsc=sc.get("dsc") or 0.0, recall=sc.get("recall"), precision=sc.get("precision"),
                     area_ratio=sc.get("area_ratio"),
                     retries=res.stage("growth").info.get("flood guard retries", 0))
        except Exception as e:
            r.update(dsc=0.0, error=repr(e)[:100])
        r["seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(r)
    return rows


def all_results():
    """Every results CSV written so far (one row per config and slice; duplicates dropped)."""
    ps = [os.path.join(OUT, n) for n in sorted(os.listdir(OUT)) if n.endswith(".csv") and n != "folds.csv"]
    d = pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)
    return d.drop_duplicates(["config", "file"])


def run_jobs(jobs_by_fold, path, workers=11, chunk=8):
    """jobs_by_fold: {fold: [(config id, arm, params)]}; runs every (config, slice) pair no file has yet."""
    folds = make_folds()
    done = _done(path, ("config", "file"))
    for n in os.listdir(OUT):
        q = os.path.join(OUT, n)
        if q != path and n.endswith(".csv") and n != "folds.csv":
            done |= _done(q, ("config", "file"))
    tasks = []
    for ip, mp, info in read_split("brats_eval", "dev"):
        fold = folds[info["file"]]
        todo = [(c, a, p, fold) for c, a, p in jobs_by_fold.get(fold, []) if (c, info["file"]) not in done]
        for i in range(0, len(todo), chunk):
            tasks.append((ip, mp, info, todo[i:i + chunk]))
    if tasks:
        print(f"{os.path.basename(path)}: {len(tasks)} tasks", flush=True)
        _drive(tasks, worker, path, FIELDS, workers, os.path.basename(path))


# ─── Scoring and decisions ────────────────────────────────────────────────────
def score_table(path, fold=None):
    return score_table_df(pd.read_csv(path), fold)


def score_table_df(d, fold=None):
    if fold:
        d = d[d.fold == fold]
    t = d.groupby(["config", "tumor"]).dsc.mean().unstack()
    t["objective"] = t[list(TYPES)].mean(axis=1)
    t["n"] = d.groupby("config").size()
    t["retries_share"] = d.groupby("config").retries.apply(lambda v: float((v > 0).mean()))
    return t


def changed(arm, p, base):
    return sum(p[k] != base[k] for k in ("sigma_cap", "k_local", "sigma_floor_min", "flood_frac"))


def pick(tab, arm, grid, base):
    """Best on fold A; ties within 0.005: fewer changes from `base`, then lower k_local."""
    best = tab.objective.max()
    near = [(c, p) for c, p in grid if c in tab.index and tab.loc[c, "objective"] >= best - TIE]
    near.sort(key=lambda cp: (changed(arm, cp[1], base), cp[1]["k_local"]))
    return near[0]


def accept(tabB, win_id, ref_id):
    w, r = tabB.loc[win_id], tabB.loc[ref_id]
    gain = float(w.objective - r.objective)
    drops = {t: float(w[t] - r[t]) for t in TYPES}
    ok = gain >= GAIN_MIN and all(v >= -TYPE_DROP_MAX for v in drops.values())
    return ok, gain, drops


def grids(round_, arm, base, prior=None):
    out = []
    if round_ == 1:
        g = R1[arm]
        for cap, kp, fl in itertools.product(g["sigma_cap"], g["kpair"], g["sigma_floor_min"]):
            p = dict(base, sigma_cap=cap, k_local=kp[0], k_global=kp[1], sigma_floor_min=fl)
            out.append((cid(arm, p), p))
    else:
        for ff, kp in itertools.product((None, prior["p95"], prior["p99"]), R2K[arm]):
            p = dict(base, flood_frac=ff, k_local=kp[0], k_global=kp[1])
            out.append((cid(arm, p), p))
    seen, uniq = set(), []
    for c, p in out:
        if c not in seen:
            seen.add(c); uniq.append((c, p))
    return uniq


def main():
    os.makedirs(OUT, exist_ok=True)
    folds = make_folds()
    prior = size_prior(folds)
    print("size prior (fold-A GT / brain):", {k: round(v, 3) for k, v in prior.items()}, flush=True)
    dec_path = os.path.join(OUT, "decisions.json")
    dec = json.load(open(dec_path)) if os.path.isfile(dec_path) else {}
    cur = {a: dict(p) for a, p in CUR.items()}

    # References on both folds: the two current arms and plain SRG.
    refs = {"A": [], "B": []}
    for fold in "AB":
        refs[fold] = [(cid(a, p), a, p) for a, p in cur.items()] + [("SRG|plain", "SRG", cur["ESRG"])]
    run_jobs(refs, os.path.join(OUT, "refs.csv"))

    for rnd in (1, 2):
        key = f"round{rnd}"
        if key in dec:                       # decided earlier: restore the accepted parameters
            for a, v in dec[key].items():
                if v["accepted"]:
                    cur[a] = v["params"]
            continue
        grid = {a: grids(rnd, a, cur[a], prior) for a in cur}
        path = os.path.join(OUT, f"round{rnd}_foldA.csv")
        run_jobs({"A": [(c, a, p) for a in grid for c, p in grid[a]]}, path)
        tabA = score_table_df(all_results(), "A")      # includes the reference rows of refs.csv
        wins = {}
        for a in cur:
            c, p = pick(tabA.loc[[c for c, _ in grid[a]]], a, grid[a], cur[a])
            wins[a] = (c, p)
        confirm_jobs = {"B": [(c, a, p) for a, (c, p) in wins.items()]}
        path_b = os.path.join(OUT, f"round{rnd}_foldB.csv")
        run_jobs(confirm_jobs, path_b)
        tabB = score_table_df(all_results(), "B")
        dec[key] = {}
        for a, (c, p) in wins.items():
            ref = cid(a, cur[a])
            ok, gain, drops = accept(tabB, c, ref) if c != ref else (False, 0.0, {})
            dec[key][a] = {"winner": c, "params": p, "accepted": bool(ok), "gain_on_B": gain, "type_changes_on_B": drops,
                           "foldA_objective": float(tabA.loc[c, "objective"]),
                           "foldA_reference": float(tabA.loc[ref, "objective"]) if ref in tabA.index else None,
                           "reference": ref, "winner_is_current": c == ref}
            if ok:
                cur[a] = p
            print(f"round {rnd} {a}: winner {c}  gain on B {gain:+.3f}  -> {'ACCEPTED' if ok else 'rejected'}", flush=True)
        json.dump(dec, open(dec_path, "w"), indent=2, default=float)

    # Final comparison on all of DEV.
    final = {a: dict(p) for a, p in cur.items()}
    allf = {"A": [(cid(a, p), a, p) for a, p in final.items()], "B": [(cid(a, p), a, p) for a, p in final.items()]}
    run_jobs(allf, os.path.join(OUT, "final.csv"))
    d = all_results()
    keep = {cid(a, p) for a, p in final.items()} | {cid(a, p) for a, p in CUR.items()} | {"SRG|plain"}
    d = d[d.config.isin(keep)]
    tab = d.groupby(["config", "tumor"]).dsc.mean().unstack()
    tab["objective"] = tab[list(TYPES)].mean(axis=1)
    tab["share_dsc_ge_0.7"] = d.groupby("config").dsc.apply(lambda v: float((v >= 0.7).mean()))
    srg = float(tab.loc["SRG|plain", "objective"])
    summ = {"final_params": final, "size_prior": prior, "decisions": dec,
            "all_dev": tab.round(4).to_dict("index"),
            "esrg_gap_to_plain_srg": srg - float(tab.loc[cid("ESRG", final["ESRG"]), "objective"]),
            "stop_rule_gap_threshold": STOP_GAP}
    summ["stop_rule_triggered"] = summ["esrg_gap_to_plain_srg"] > STOP_GAP
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)
    base = Config.load(FROZEN)
    e, s = final["ESRG"], final["SRG+S"]
    v2 = base.replace(k_local=e["k_local"], k_global=e["k_global"], sigma_cap=e["sigma_cap"],
                      sigma_floor_min=e["sigma_floor_min"], flood_frac=e["flood_frac"],
                      srg_k_local=s["k_local"], srg_k_global=s["k_global"])
    v2.save(FROZEN_V2)
    print(json.dumps({k: summ[k] for k in ("final_params", "esrg_gap_to_plain_srg", "stop_rule_triggered")}, indent=2, default=float))
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
