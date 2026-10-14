"""
figures_brats.py — Figures of Chapter 4 (BraTS 2023 evaluation).

Purpose : Draw the figure of the results chapter from a results folder of
          experiments/evaluate.py (raw_results.csv, manual_seeds.csv).
Function : fig_examples() draws one typical slice per tumor type (the slice whose P_SRG DSC is closest
          to the tumor type's median), re-run with the logged seeds so that the masks equal the evaluated
          ones (checked by mask hash). Figure 4.1 = the four overall configurations (one-seed SRG, planted and
          automatic ESRG, three-seed SRG as reference); Figure 4.2 = the effect of each applied method.
          CLI: python -P experiments/chapter4/figures_brats.py <results folder> <out folder>
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esrg import run
from esrg.visualize import outline
from experiments.evaluate import CONFIGS, base_config
from experiments.sampling import BRATS_OUT, read_split

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11})
TYPES = [("glioma", "Glioma"), ("meningioma", "Meningioma")]
FIG1 = [("P_SRG_solo", "SRG\n(1 planted seed)"), ("P_ESRG", "ESRG\n(1 planted seed)"),
        ("A_ESRG", "ESRG\n(automatic seed)"), ("P_SRG", "SRG, 3 seeds\n(reference)")]
FIG2 = [("P_SRG_L", "SRG + LLDM\n(3 seeds)"), ("P_ESRG_noL", "ESRG\nwithout LLDM"),
        ("P_SRG_solo_S", "SRG + ASC\n(1 seed)"), ("P_ESRG_C_noS", "ESRG\nwithout ASC")]


def load(d):
    df = pd.read_csv(os.path.join(d, "raw_results.csv"), low_memory=False)
    return df[df.error.isna()] if "error" in df else df


def _typical(df, t):
    p = df[(df.config == "P_SRG") & (df.tumor == t)].dropna(subset=["dsc"])
    med = p.dsc.median()
    return p.iloc[(p.dsc - med).abs().argsort().iloc[0]].file


def fig_examples(df, d, set_name, out, columns):
    seeds = pd.read_csv(os.path.join(d, "manual_seeds.csv"), header=0, names=["file", "placement", "points", "rule"])
    paths = {info["file"]: (ip, mp) for ip, mp, info in read_split(BRATS_OUT, set_name)}
    base = base_config()
    picked = [(t, name, _typical(df, t)) for t, name in TYPES]
    fig, axs = plt.subplots(2, len(columns) + 1, figsize=(2.3 * (len(columns) + 1), 5.6))
    checks = []
    for row, (t, name, file) in enumerate(picked):
        ip, mp = paths[file]
        pts = {int(r.placement): [tuple(p) for p in json.loads(r.points)] for r in seeds[seeds.file == file].itertuples()}
        crop = None
        for col, (cid, label) in enumerate([(None, "Slice and\nground truth")] + columns):
            ax = axs[row, col]; ax.axis("off")
            if cid is None:
                res = run(ip, base.replace(seed_mode="auto"), mask_path=mp)
                img = res.stage("input").image; gt = res.gt.astype(bool)
                rr, cc = np.nonzero(res.stage("mask").image)
                crop = (slice(max(rr.min() - 8, 0), rr.max() + 8), slice(max(cc.min() - 8, 0), cc.max() + 8))
                m, txt = gt, label
            else:
                _, over, seeding, placement, types = CONFIGS[cid]
                res = run(ip, base.replace(seed_mode=seeding, **over), mask_path=mp,
                          manual_points=[p for p in pts[placement] if p[2] in types])
                m = res.mask.astype(bool)
                logged = df[(df.file == file) & (df.config == cid)].mask_sha1.iloc[0]
                checks.append(bool(res.scores["mask_sha1"] == logged))
                txt = f"{label}\nDSC = {res.scores['dsc']:.2f}"
            rgb = np.stack([img] * 3, -1) / 255.0
            ov = rgb.copy()
            ov[m] = 0.55 * ov[m] + 0.45 * np.array([0.85, 0.1, 0.1])
            ov[outline(m, 1)] = [0.95, 0.0, 0.0]
            if cid is not None:
                ov[outline(gt, 1) & ~outline(m, 1)] = [1.0, 0.85, 0.0]
            ax.imshow(ov[crop]); ax.set_title(txt, fontsize=12)
        axs[row, 0].text(-0.08, 0.5, name, transform=axs[row, 0].transAxes, rotation=90, va="center", ha="right", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=200, facecolor="white"); plt.close(fig)
    return [p[2] for p in picked], checks


def main(d, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = load(d)
    set_name = df.set.iloc[0]
    files, c1 = fig_examples(df, d, set_name, os.path.join(out_dir, "figure_4_1.png"), FIG1)
    _, c2 = fig_examples(df, d, set_name, os.path.join(out_dir, "figure_4_2.png"), FIG2)
    checks = c1 + c2
    json.dump({"examples": files, "mask_hash_matches": checks}, open(os.path.join(out_dir, "figures.json"), "w"), indent=1)
    print("figures written; examples:", files, "| hash matches:", checks)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
