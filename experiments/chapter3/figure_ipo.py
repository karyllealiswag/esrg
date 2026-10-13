"""
figure_ipo.py — Figure 3.1: Input-Process-Output structure of the proposed system (BraTS 2023 version).

The example is a DEVELOPMENT-set meningioma slice whose automatic-ESRG DSC equals the median of
the rehearsal (D3) meningioma results, so it is typical rather than selected for quality. No TEST
slice is used. Run: python -P experiments/chapter3/figure_ipo.py <out.png>
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from esrg import Config, run
from esrg.visualize import outline

FILE = "brats_eval/dev/images/meningioma_BraTS-MEN-00424-000_ax_113_t1c.png"
CFG = "configs/esrg_brats_frozen_v3.json"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})


def box(ax, x, y, w, h, fc="white", lw=1.4, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec="black", lw=lw))


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"))


def stage(ax, x, y, title, body):
    box(ax, x, y, 0.118, 0.16, lw=1.1, r=0.008)
    ax.text(x + 0.059, y + 0.142, title, ha="center", va="top", fontsize=9.5, fontweight="bold", wrap=True)
    ax.text(x + 0.059, y + 0.075, body, ha="center", va="center", fontsize=8.2)


def main(out):
    res = run(FILE, Config.load(CFG).replace(seed_mode="auto"))
    img = res.stage("input").image
    over = np.stack([img] * 3, -1) / 255.0
    m = res.mask.astype(bool)
    over[m] = 0.55 * over[m] + 0.45 * np.array([0.85, 0.1, 0.1])
    over[outline(m, 2)] = [0.9, 0.0, 0.0]
    # crop both panels to the brain (plus a margin) so the slice is legible at figure size
    rr, cc = np.nonzero(res.stage("mask").image)
    r0, r1 = max(rr.min() - 12, 0), min(rr.max() + 12, img.shape[0])
    c0, c1 = max(cc.min() - 12, 0), min(cc.max() + 12, img.shape[1])
    img, over = img[r0:r1, c0:c1], over[r0:r1, c0:c1]

    fig = plt.figure(figsize=(16, 9.4), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # INPUT
    box(ax, 0.008, 0.18, 0.19, 0.64)
    ax.text(0.103, 0.78, "INPUT", ha="center", fontsize=16, fontweight="bold")
    ia = fig.add_axes([0.033, 0.43, 0.14, 0.30]); ia.imshow(img, cmap="gray"); ia.axis("off")
    ax.text(0.103, 0.385, "Skull-stripped, axial contrast-enhanced\nT1-weighted 2D MRI slice (PNG)", ha="center", va="top", fontsize=10)
    ax.text(0.103, 0.30, "Frozen parameter configuration\n(Table 3.5)", ha="center", va="top", fontsize=10)
    ax.text(0.103, 0.235, "Manual seed clicks\n(optional)", ha="center", va="top", fontsize=10, style="italic")
    arrow(ax, 0.2, 0.5, 0.232, 0.5)

    # PROCESS
    box(ax, 0.238, 0.03, 0.455, 0.94)
    ax.text(0.465, 0.935, "PROCESS", ha="center", fontsize=16, fontweight="bold")
    rows = [(0.70, "Pre-processing", [("Resizing and\nNormalization", "longer side ≤ 512 px,\n0.5th–99.5th percentiles"),
                                       ("Brain Masking", "largest non-zero\nregion, hole filling"),
                                       ("Log-Domain\nTransformation", "noise floor $\\sigma_{floor}$\n(Objective 2)")]),
            (0.385, "Core Processing", [("Automated Seed\nSelection", "interior multi-level Otsu,\nranking, seed core\n(Objective 1)"),
                                        ("Region Growing", "local log-domain $\\delta$\n(Objective 2)"),
                                        ("Adaptive Stopping\nCriterion", "$T_L$, $T_G$ around the running\nregion mean; $\\sigma$ cap,\nflood guard (Objective 3)")]),
            (0.07, "Post-processing", [("Hole Filling", "seals necrotic core"),
                                       ("Morphological\nOpening", "disk of radius 1 px"),
                                       ("Seed-Connected\nComponent", "keeps component(s)\ncontaining the seed")])]
    for y, title, stages in rows:
        box(ax, 0.256, y, 0.42, 0.225, fc="#f2f2f2", lw=1.1, r=0.012)
        ax.text(0.466, y + 0.2, title, ha="center", fontsize=12.5, fontweight="bold")
        for i, (t, b) in enumerate(stages):
            x = 0.27 + i * 0.137
            stage(ax, x, y + 0.012, t, b)
            if i < 2:
                arrow(ax, x + 0.119, y + 0.09, x + 0.136, y + 0.09)
    arrow(ax, 0.466, 0.695, 0.466, 0.615)
    arrow(ax, 0.466, 0.38, 0.466, 0.3)
    arrow(ax, 0.695, 0.68, 0.73, 0.68)

    # OUTPUT
    box(ax, 0.735, 0.44, 0.255, 0.53)
    ax.text(0.862, 0.925, "OUTPUT", ha="center", fontsize=16, fontweight="bold")
    oa = fig.add_axes([0.79, 0.63, 0.145, 0.26]); oa.imshow(over); oa.axis("off")
    ax.text(0.862, 0.6, "Binary tumor mask and overlay,\nstatus flag (OK, WARN, or\nNO TUMOR CANDIDATE), and\nstage-by-stage intermediates",
            ha="center", va="top", fontsize=10)
    arrow(ax, 0.862, 0.438, 0.862, 0.408)

    # EVALUATION
    box(ax, 0.735, 0.03, 0.255, 0.375)
    ax.text(0.862, 0.365, "EVALUATION", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.862, 0.325, "Ground-truth tumor-core mask (PNG), used only here", ha="center", fontsize=9.5, style="italic")
    y = 0.28
    for head, body in (("Overlap:", "DSC, Precision, Recall"), ("Boundary:", "HD95"),
                       ("Objective-specific:", "Seed Hit Rate,\nSeed-placement SD of DSC")):
        ax.text(0.75, y, head, fontsize=10.5, fontweight="bold", va="top")
        ax.text(0.75, y - 0.035, body, fontsize=10, va="top")
        y -= 0.085 if "\n" not in body else 0.11
    fig.savefig(out, dpi=170, facecolor="white")
    print("wrote", out, "| example DSC (auto ESRG):", round(res.scores.get("dsc", float("nan")), 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/chapter3/figure_3_1.png")
