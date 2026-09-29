"""
test_pipeline.py — Smoke and correctness tests.

Purpose : Guard the metric math, normalization, growth stopping, and config rules.
Function : Small synthetic cases with known answers — perfect/disjoint DSC, output
          range, a bright square the grower must fill without leaking, seed-hit
          logic, rejection of an invalid k_global <= k_local, and the seed-pixel
          report (per-region grouping, averages, click de-duplication).
Notes   : Run with pytest, or directly: python tests/test_pipeline.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrg import Config
from esrg.metrics import evaluate, seed_hit
from esrg.growing import grow_esrg
from esrg.preprocessing import normalize, log_transform
from esrg.pixel_report import manual_groups, auto_groups, describe


def test_metrics_perfect_and_disjoint():
    gt = np.zeros((20, 20), bool); gt[5:15, 5:15] = True
    assert evaluate(gt, gt)["dsc"] == 1.0
    other = np.zeros((20, 20), bool); other[0:3, 0:3] = True
    assert evaluate(other, gt)["dsc"] == 0.0


def test_normalize_range():
    a = normalize(np.random.rand(30, 30) * 4000)
    assert a.min() >= 0 and a.max() <= 255


def test_growing_stops_at_a_hard_edge():
    """A bright square on a dark field must not leak into the background."""
    img = np.full((60, 60), 40.0); img[20:40, 20:40] = 200.0
    img += np.random.RandomState(0).normal(0, 2, img.shape)
    L = log_transform(normalize(img))
    mask = np.ones((60, 60), bool)
    core = np.zeros((60, 60), bool); core[28:32, 28:32] = True
    region, trace = grow_esrg(L, mask, core, 0.02, Config())
    truth = np.zeros((60, 60), bool); truth[20:40, 20:40] = True
    assert evaluate(region, truth)["dsc"] > 0.9, trace
    assert region.sum() < 0.5 * region.size


def test_seed_hit_logic():
    gt = np.zeros((10, 10), bool); gt[2:8, 2:8] = True
    inside = np.zeros((10, 10), bool); inside[4:6, 4:6] = True
    outside = np.zeros((10, 10), bool); outside[0, 0] = True
    assert seed_hit(inside, gt) is True and seed_hit(outside, gt) is False


def test_config_rejects_bad_multipliers():
    try:
        Config().replace(k_local=3.0, k_global=2.0)
    except AssertionError:
        return
    raise AssertionError("k_global <= k_local should be rejected")


def _ramp():
    raw = np.arange(25, dtype=float).reshape(5, 5) * 10        # raw[r, c] = 10 * (5r + c)
    return raw, raw / 2


def test_pixel_report_single_pixel_has_exact_value():
    raw, norm = _ramp()
    (g,) = describe(manual_groups([(2, 3, 1)]), raw, norm)
    assert g["label"] == "Region 1 (tumor)" and g["n"] == 1
    assert g["pixels"] == [{"row": 2, "col": 3, "raw": 130.0, "norm": 65.0}]
    assert g["mean_raw"] == 130.0 and g["mean_norm"] == 65.0


def test_pixel_report_groups_regions_and_averages_each():
    raw, norm = _ramp()
    pts = [(4, 4, 2), (0, 0, 1), (0, 2, 1), (1, 1, 2)]
    g1, g2 = describe(manual_groups(pts), raw, norm)
    assert [g1["label"], g2["label"]] == ["Region 1 (tumor)", "Region 2"]
    assert [(p["row"], p["col"]) for p in g1["pixels"]] == [(0, 0), (0, 2)]   # click order kept
    assert g1["mean_raw"] == 10.0 and g1["mean_norm"] == 5.0                   # (0 + 20) / 2
    assert g2["mean_raw"] == 150.0 and g2["mean_norm"] == 75.0                 # (240 + 60) / 2


def test_pixel_report_repeat_click_counts_once():
    raw, norm = _ramp()
    (g,) = describe(manual_groups([(1, 1, 1), (1, 1, 1), (3, 3, 1)]), raw, norm)
    assert g["n"] == 2 and g["mean_raw"] == (60.0 + 180.0) / 2


def test_pixel_report_auto_core_is_row_major_and_empty_is_none():
    raw, norm = _ramp()
    core = np.zeros((5, 5), bool); core[3, 1] = core[1, 4] = core[1, 2] = True
    (g,) = describe(auto_groups(core), raw, norm)
    assert g["label"] == "System seed core"
    assert [(p["row"], p["col"]) for p in g["pixels"]] == [(1, 2), (1, 4), (3, 1)]
    assert g["mean_raw"] == (70.0 + 90.0 + 160.0) / 3
    assert auto_groups(np.zeros((5, 5), bool)) == [] and manual_groups([]) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
