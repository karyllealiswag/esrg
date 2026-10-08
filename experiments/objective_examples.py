"""
objective_examples.py — Example slices where ESRG beats SRG, one set per specific objective.

Purpose : Pick, re-run and render example slices for the Word comparison document, so
          every number and picture can be counter-checked against raw_results.csv.
Function : select() ranks slices from the evaluation (test split first) for each
          objective; render() re-runs the pipeline for every image shown, asserts the
          DSC equals the value recorded in raw_results.csv, and writes one PNG per
          image (red = prediction, green = ground-truth outline, as in the GUI) plus
          examples.json describing every example.
Notes   : Objective 1 — 3 SRG operator clicks vs ESRG automatic seeding.
          Objective 2 — 3 SRG operator clicks vs ESRG from operator click 1 (same seed).
          Objective 3 — SRG vs ESRG without the stopping criterion vs full ESRG, all from
          the planted seed (SRG itself does not leak here; it undersegments).
          CLI: python -m experiments.objective_examples [--per-objective 3] [--out outputs/examples]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esrg import Config, run
from esrg.io_utils import load_image, load_mask
from esrg.visualize import overlay_result
from experiments.evaluate import CONFIGS, N_CLICKS, _disk, operator_clicks, planted_core
from experiments.sampling import read_manifest

EVAL = "outputs/evaluation"
TUMOR_ORDER = ["meningioma", "pituitary", "glioma"]

# objective -> (heading, SRG configs, ESRG configs); a tile is (label, config id)
OBJECTIVES = {
    1: dict(
        title="Objective 1: Automatic seed selection",
        srg=[("SRG, operator click %d" % k, f"O_SRG_{k}") for k in (1, 2, 3)],
        esrg=[("ESRG, automatic seed", "A_ESRG")]),
    2: dict(
        title="Objective 2: Log-domain transformation (undersegmentation)",
        srg=[("SRG, operator click %d" % k, f"O_SRG_{k}") for k in (1, 2, 3)],
        esrg=[("ESRG, operator click 1", "O_ESRG_1")]),
    3: dict(
        title="Objective 3: Adaptive stopping criterion (boundary leakage)",
        srg=[("SRG, planted seed", "P_SRG"), ("ESRG, stopping criterion OFF", "P_ESRG_nostop")],
        esrg=[("ESRG, full (stopping criterion ON)", "P_ESRG")]),
}


def _wide(d, cfgs, col):
    return d[d.config.isin(cfgs)].pivot_table(index="file", columns="config", values=col)


def select(per_objective=3):
    """{objective: [file, ...]}, best cases first, test split first, tumor classes mixed."""
    d = pd.read_csv(os.path.join(EVAL, "raw_results.csv"), low_memory=False)
    meta = d[d.config == "A_ESRG"].set_index("file")[["split", "tumor", "plane"]]
    dsc = _wide(d, list(CONFIGS), "dsc")
    rec = _wide(d, list(CONFIGS), "recall")
    pre = _wide(d, ["P_ESRG_nostop"], "precision")
    srg3 = ["O_SRG_1", "O_SRG_2", "O_SRG_3"]
    cand = {
        1: (dsc.A_ESRG >= 0.8) & (dsc[srg3].max(axis=1) <= 0.3),
        2: (dsc.O_ESRG_1 >= 0.8) & (rec[srg3].max(axis=1) <= 0.3),
        3: (dsc.P_ESRG >= 0.85) & (pre.P_ESRG_nostop <= 0.4) & (dsc.P_SRG <= 0.4),
    }
    score = {
        1: dsc.A_ESRG - dsc[srg3].max(axis=1),
        2: dsc.O_ESRG_1 - dsc[srg3].max(axis=1),
        3: dsc.P_ESRG - dsc.P_ESRG_nostop,
    }
    used, out = set(), {}
    for obj in (1, 2, 3):
        c = pd.DataFrame({"score": score[obj][cand[obj]]}).join(meta)
        c = c[~c.index.isin(used)].sort_values("score", ascending=False)
        c["rank_split"] = (c.split != "test").astype(int)        # test split first
        c = c.sort_values(["rank_split", "score"], ascending=[True, False])
        picks = []
        for tumor in TUMOR_ORDER:                               # one per tumor class first
            t = c[(c.tumor == tumor) & ~c.index.isin(picks)]
            if len(t) and len(picks) < per_objective:
                picks.append(t.index[0])
        for f in c.index:                                       # then fill with the best left
            if len(picks) >= per_objective:
                break
            if f not in picks:
                picks.append(f)
        out[obj] = picks
        used.update(picks)
    return out


def run_config(ip, mp, name, cid):
    """Re-run one configuration exactly as experiments/evaluate.py process() does."""
    base = Config()
    img, _ = load_image(ip, base.max_side)
    gt = load_mask(mp, img.shape)
    _, over, seeding, _, click = CONFIGS[cid]
    cfg = base.replace(**over)
    core, click_rc = None, None
    if seeding == "planted":
        core, (r, c, _) = planted_core(gt, base)
        click_rc = (r, c)
    elif seeding == "operator":
        r, c, _ = operator_clicks(gt, base, name)[(click - 1) % N_CLICKS]
        core, click_rc = _disk(gt.shape, r, c, base.manual_seed_radius), (r, c)
    res = run(ip, cfg, mask_path=mp, planted_core=core)
    return img, gt, res, click_rc


def render(out_dir, per_objective=3):
    d = pd.read_csv(os.path.join(EVAL, "raw_results.csv"), low_memory=False)
    d = d.set_index(["file", "config"])
    paths = {r["file"]: (ip, mp, r) for ip, mp, r in read_manifest(EVAL)}
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for obj, files in select(per_objective).items():
        spec = OBJECTIVES[obj]
        for f in files:
            ip, mp, info = paths[f]
            ex = {"objective": obj, "title": spec["title"], "file": f, "split": info["split"],
                  "tumor": info["tumor"], "plane": info["plane"], "image_path": ip,
                  "mask_path": mp, "srg": [], "esrg": []}
            for group in ("srg", "esrg"):
                for label, cid in spec[group]:
                    img, gt, res, click_rc = run_config(ip, mp, f, cid)
                    rec = d.loc[(f, cid)]
                    dsc = float(res.scores["dsc"])
                    assert abs(dsc - float(rec.dsc)) < 1e-3, (f, cid, dsc, float(rec.dsc))
                    png = f"obj{obj}_{f[:-4]}_{cid}.png"
                    Image.fromarray(overlay_result(img, res.mask, gt)).save(os.path.join(out_dir, png))
                    ex[group].append({
                        "label": label, "config": cid, "png": png, "dsc": round(dsc, 3),
                        "precision": round(float(res.scores["precision"]), 3),
                        "recall": round(float(res.scores["recall"]), 3),
                        "click_row_col": list(click_rc) if click_rc else None,
                        "pred_area": int(res.mask.sum()), "gt_area": int(gt.sum())})
                    print(f"obj{obj} {f} {cid:15s} DSC {dsc:.3f}  (csv {float(rec.dsc):.3f})", flush=True)
            manifest.append(ex)
    with open(os.path.join(out_dir, "examples.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-objective", type=int, default=3)
    ap.add_argument("--out", default="outputs/examples")
    ap.add_argument("--select-only", action="store_true", help="print the chosen files and stop")
    a = ap.parse_args()
    if a.select_only:
        for o, fs in select(a.per_objective).items():
            print(o, fs)
    else:
        render(a.out, a.per_objective)
