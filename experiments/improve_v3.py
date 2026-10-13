"""
improve_v3.py — The pre-declared ESRG improvement search v3 (experiments/IMPROVEMENT_PROTOCOL_ESRG_V3.md).

Purpose : Test the running global mean and the healthy-tissue rule on ESRG (standalone engine)
          on the BraTS DEVELOPMENT set only, and decide each round by the protocol's fixed rule.
Function : make_split() draws D1 (50%, screen) / D2 (25%, confirm) / D3 (25%, final) per tumor
          type; rounds() runs Round 1 (20 configurations) and Round 2 (10) on D1, confirms each
          winner on D2 and applies the acceptance rule; it stops at the review point (§4). The
          SRG mirrors and the D3 check (§5–6) are run later with --stage mirrors / --stage final.
Notes   : Resumable and crash-safe (rows appended and flushed; rerunning continues). Planted P1
          seeds from experiments/evaluate.py. Starts from configs/esrg_brats_frozen_v2.json and
          never overwrites it. TEST is never read.
          CLI: python -P experiments/improve_v3.py --stage rounds
               python -P experiments/improve_v3.py --stage mirrors   (after the review point)
               python -P experiments/improve_v3.py --stage final     (D3, looked at once)
"""
import argparse
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
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg.io_utils import load_image, load_mask
from experiments.evaluate import seed_protocol
from experiments.sampling import read_split
from experiments.tune import _done, _drive

OUT = "outputs/improve_v3"
V2 = "configs/esrg_brats_frozen_v2.json"
V3 = "configs/esrg_brats_frozen_v3.json"
TYPES = ("glioma", "meningioma")
GAIN_MIN, TYPE_DROP_MAX, TIE, SUCCESS = 0.02, 0.01, 0.005, 0.02
FIELDS = ["config", "arm", "set", "file", "tumor", "dsc", "recall", "precision", "area_ratio", "retries",
          "seconds", "error"]
KEYS = ("global_running", "healthy_rule", "healthy_proto", "addons", "k_local", "k_global")
CUR = dict(global_running=False, healthy_rule="off", healthy_proto="median", addons=True, k_local=4.0, k_global=5.0)


def cid(arm, p):
    if arm != "ESRG":
        return arm
    return (f"ESRG|run={int(p['global_running'])}|rule={p['healthy_rule']}|proto={p['healthy_proto']}"
            f"|addons={int(p['addons'])}|kl={p['k_local']:g}|kg={p['k_global']:g}")


def to_config(arm, p, v2):
    """Configs of the arms. ESRG reads p; the SRG arms use ESRG's settings (no separate tuning)."""
    if arm == "SRG|8conn":
        return v2.replace(method="srg", seed_mode="manual", srg_connectivity=8)
    if arm == "SRG|4conn":
        return v2.replace(method="srg", seed_mode="manual", srg_connectivity=4)
    if arm.startswith("SRG+"):       # mirrors: ESRG's accepted settings for that method, none tuned for SRG
        common = dict(method="srg", seed_mode="manual", srg_connectivity=8, srg_k_local=None, srg_k_global=None,
                      srg_local_radius=None, k_local=p["k_local"], k_global=p["k_global"],
                      sigma_cap=v2.sigma_cap if p["addons"] else None,
                      flood_frac=v2.flood_frac if p["addons"] else None, healthy_rule=p["healthy_rule"],
                      healthy_proto=p["healthy_proto"])
        flag = {"SRG+LLDM": dict(srg_use_log_local=True), "SRG+ASC": dict(srg_use_stopping=True),
                "SRG+HEALTHY": dict(srg_use_healthy=True)}[arm]
        if arm != "SRG+HEALTHY":
            common["healthy_rule"] = "off"
        return v2.replace(**common, **flag)
    return v2.replace(seed_mode="manual", global_running=p["global_running"], healthy_rule=p["healthy_rule"],
                      healthy_proto=p["healthy_proto"], k_local=p["k_local"], k_global=p["k_global"],
                      sigma_cap=v2.sigma_cap if p["addons"] else None,
                      flood_frac=v2.flood_frac if p["addons"] else None)


# ─── Split ────────────────────────────────────────────────────────────────────
def make_split():
    path = os.path.join(OUT, "split.csv")
    if os.path.isfile(path):
        return dict(csv.reader(open(path)))
    out = {}
    for t in TYPES:
        files = sorted(r["file"] for _, _, r in read_split("brats_eval", "dev") if r["tumor"] == t)
        random.Random(2026).shuffle(files)
        n1, n2 = round(0.50 * len(files)), round(0.75 * len(files))
        out.update({f: ("D1" if i < n1 else "D2" if i < n2 else "D3") for i, f in enumerate(files)})
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(sorted(out.items()))
    return out


# ─── Worker ───────────────────────────────────────────────────────────────────
def worker(task):
    ip, mp, info, jobs = task
    v2 = Config.load(V2)
    raw, _ = load_image(ip, v2.max_side)
    gt = load_mask(mp, raw.shape)
    seeds, _ = seed_protocol(raw, gt, v2, info["file"])
    rows = []
    for c, arm, p, st in jobs:
        r = {"config": c, "arm": arm, "set": st, "file": info["file"], "tumor": info["tumor"]}
        t0 = time.perf_counter()
        try:
            res = run(ip, to_config(arm, p, v2), mask_path=mp, manual_points=seeds[1])
            sc = res.scores
            r.update(dsc=sc.get("dsc") or 0.0, recall=sc.get("recall"), precision=sc.get("precision"),
                     area_ratio=sc.get("area_ratio"),
                     retries=res.stage("growth").info.get("flood guard retries", 0) if res.stage("growth") else 0)
        except Exception as e:
            r.update(dsc=0.0, error=repr(e)[:100])
        r["seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(r)
    return rows


def run_jobs(jobs_by_set, path, workers=11, chunk=6):
    split = make_split()
    done = set()
    for n in os.listdir(OUT):
        if n.endswith(".csv") and n != "split.csv":
            done |= _done(os.path.join(OUT, n), ("config", "file"))
    tasks = []
    for ip, mp, info in read_split("brats_eval", "dev"):
        st = split[info["file"]]
        todo = [(c, a, p, st) for c, a, p in jobs_by_set.get(st, []) if (c, info["file"]) not in done]
        for i in range(0, len(todo), chunk):
            tasks.append((ip, mp, info, todo[i:i + chunk]))
    if tasks:
        print(f"{os.path.basename(path)}: {len(tasks)} tasks", flush=True)
        _drive(tasks, worker, path, FIELDS, workers, os.path.basename(path))


def results():
    ps = [os.path.join(OUT, n) for n in sorted(os.listdir(OUT)) if n.endswith(".csv") and n != "split.csv"]
    return pd.concat([pd.read_csv(p) for p in ps], ignore_index=True).drop_duplicates(["config", "file"])


def table(d, st):
    d = d[d.set == st]
    t = d.groupby(["config", "tumor"]).dsc.mean().unstack()
    t["objective"] = t[list(TYPES)].mean(axis=1)
    t["n"] = d.groupby("config").size()
    return t


# ─── Rounds ───────────────────────────────────────────────────────────────────
def grid(rnd, base):
    out = []
    if rnd == 1:
        for run_, rule, add in itertools.product((False, True), ("off", "veto", "rescue", "both"), (True, False)):
            out.append(dict(base, global_running=run_, healthy_rule=rule, addons=add))
        for proto, rule in itertools.product(("mode", "otsu3"), ("veto", "both")):
            out.append(dict(base, global_running=False, healthy_rule=rule, healthy_proto=proto, addons=True))
    else:
        for (kl, kg), add in itertools.product(((2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 8.0)), (True, False)):
            out.append(dict(base, k_local=kl, k_global=kg, addons=add))
    seen, uniq = set(), []
    for p in out:
        if cid("ESRG", p) not in seen:
            seen.add(cid("ESRG", p)); uniq.append(p)
    return uniq


def changes(p, base):
    return sum(p[k] != base[k] for k in KEYS)


def rounds():
    os.makedirs(OUT, exist_ok=True)
    make_split()
    dec_path = os.path.join(OUT, "decisions.json")
    dec = json.load(open(dec_path)) if os.path.isfile(dec_path) else {}
    cur = dict(CUR)
    refs = [("SRG|8conn", "SRG|8conn", cur), ("SRG|4conn", "SRG|4conn", cur), (cid("ESRG", cur), "ESRG", cur)]
    run_jobs({"D1": refs, "D2": refs}, os.path.join(OUT, "refs.csv"))
    for rnd in (1, 2):
        key = f"round{rnd}"
        if key in dec:
            if dec[key]["accepted"]:
                cur = dec[key]["params"]
            continue
        g = grid(rnd, cur)
        run_jobs({"D1": [(cid("ESRG", p), "ESRG", p) for p in g]}, os.path.join(OUT, f"round{rnd}_D1.csv"))
        t1 = table(results(), "D1")
        best = t1.loc[[cid("ESRG", p) for p in g], "objective"].max()
        near = [p for p in g if t1.loc[cid("ESRG", p), "objective"] >= best - TIE]
        near.sort(key=lambda p: (changes(p, cur), p["k_local"]))
        win = near[0]
        wid, rid = cid("ESRG", win), cid("ESRG", cur)
        run_jobs({"D2": [(wid, "ESRG", win)]}, os.path.join(OUT, f"round{rnd}_D2.csv"))
        t2 = table(results(), "D2")
        gain = float(t2.loc[wid, "objective"] - t2.loc[rid, "objective"])
        drops = {t: float(t2.loc[wid, t] - t2.loc[rid, t]) for t in TYPES}
        ok = wid != rid and gain >= GAIN_MIN and all(v >= -TYPE_DROP_MAX for v in drops.values())
        dec[key] = {"winner": wid, "params": win, "reference": rid, "accepted": bool(ok),
                    "D1_winner": float(t1.loc[wid, "objective"]), "D1_reference": float(t1.loc[rid, "objective"]),
                    "D2_gain": gain, "D2_type_changes": drops, "n_tied_on_D1": len(near)}
        json.dump(dec, open(dec_path, "w"), indent=2, default=float)
        print(f"round {rnd}: winner {wid}  D1 {t1.loc[wid,'objective']:.3f} (ref {t1.loc[rid,'objective']:.3f})"
              f"  D2 gain {gain:+.3f} -> {'ACCEPTED' if ok else 'rejected'}", flush=True)
        if ok:
            cur = win
    json.dump({"current_after_rounds": cur}, open(os.path.join(OUT, "state.json"), "w"), indent=2)
    d = results()
    for st in ("D1", "D2"):
        t = table(d, st)
        keep = [c for c in ("SRG|8conn", "SRG|4conn", cid("ESRG", CUR), cid("ESRG", cur)) if c in t.index]
        print(f"\n{st} (mean DSC):\n{t.loc[keep].round(3).to_string()}")
    print("\nReview point reached (protocol §4): SRG mirrors and the D3 check wait for the user.")


def mirrors():
    cur = json.load(open(os.path.join(OUT, "state.json")))["current_after_rounds"]
    arms = [(a, a, cur) for a in ("SRG+LLDM", "SRG+ASC")] + ([("SRG+HEALTHY", "SRG+HEALTHY", cur)]
                                                             if cur["healthy_rule"] != "off" else [])
    run_jobs({"D1": arms, "D2": arms}, os.path.join(OUT, "mirrors.csv"))
    d = results()
    for st in ("D1", "D2"):
        t = table(d, st)
        keep = [c for c in ("SRG|8conn", "SRG+LLDM", "SRG+ASC", "SRG+HEALTHY", cid("ESRG", cur)) if c in t.index]
        print(f"\n{st} (mean DSC):\n{t.loc[keep].round(3).to_string()}")


def final():
    cur = json.load(open(os.path.join(OUT, "state.json")))["current_after_rounds"]
    arms = [("SRG|8conn", "SRG|8conn", cur), (cid("ESRG", cur), "ESRG", cur)]
    run_jobs({"D3": arms}, os.path.join(OUT, "final_D3.csv"))
    t = table(results(), "D3")
    e, s = t.loc[cid("ESRG", cur)], t.loc["SRG|8conn"]
    margin = float(e.objective - s.objective)
    out = {"esrg": cur, "D3": t.loc[[cid("ESRG", cur), "SRG|8conn"]].round(4).to_dict("index"),
           "esrg_minus_srg": margin, "success": margin >= SUCCESS}
    json.dump(out, open(os.path.join(OUT, "final.json"), "w"), indent=2, default=float)
    v2 = Config.load(V2)
    to_config("ESRG", cur, v2).replace(seed_mode="auto").save(V3)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("rounds", "mirrors", "final"), required=True)
    a = ap.parse_args()
    {"rounds": rounds, "mirrors": mirrors, "final": final}[a.stage]()
