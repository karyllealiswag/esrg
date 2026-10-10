"""
test_pipeline.py — Smoke and correctness tests.

Purpose : Guard the metric math, normalization, growth stopping, and config rules.
Function : Small synthetic cases with known answers — perfect/disjoint DSC, output
          range, a bright square the grower must fill without leaking, seed-hit
          logic, rejection of an invalid k_global <= k_local, the seed-pixel
          report (per-region grouping, averages, click de-duplication), the
          evaluation pipeline (Cochran sample size, equal allocation, the
          statistics, worst-case distances), the GUI's metric explanations, and
          the stage telemetry (growth recording changes nothing and every logged
          decision obeys Eq. 3.8–3.11; every stage has a report), and the SRG
          baseline with ESRG modules applied (off = the published rule; with
          termination every region-1 decision obeys T_L and T_G).
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


def test_cochran_sample_size_matches_hand_computation():
    from experiments.sampling import cochran, margin
    r = cochran(1757)                         # pituitary population of the pooled splits
    assert abs(r["n0"] - 384.1459) < 1e-3     # 1.959964^2 * 0.25 / 0.05^2
    assert r["n_min"] == 316                  # 384.146 / (1 + 383.146 / 1757) = 315.37
    assert abs(margin(318, 1757) - 0.04975) < 1e-4


def test_equal_allocation_gives_every_stratum_the_same_size():
    from experiments.sampling import plan
    cells = {(t, p): [(f"{t}{p}{i}.jpg", f"{t}{p}{i}.png") for i in range(400 + 7 * k)]
             for k, (t, p) in enumerate([(t, p) for t in ("glioma", "meningioma", "pituitary")
                                         for p in ("axial", "coronal", "sagittal")])}
    pl = plan(cells)
    assert pl["n"] == 9 * pl["n_h"]
    assert pl["n_h"] * 3 >= pl["n_class_required"]


def test_wilcoxon_matches_scipy_and_holm_is_monotone():
    from scipy import stats as st
    from experiments.stats import wilcoxon, holm
    rng = np.random.RandomState(1)
    x = rng.rand(80); y = x + rng.normal(0.05, 0.1, 80)
    w = wilcoxon(x, y)
    ref = st.wilcoxon(y - x, correction=False, method="approx")
    assert abs(w["p"] - ref.pvalue) < 1e-10
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_wilson_interval_known_value():
    from experiments.stats import wilson
    lo, hi = wilson(50, 100)                  # p = .5, n = 100 -> [40.38, 59.62]
    assert abs(lo - 40.383) < 1e-2 and abs(hi - 59.617) < 1e-2


def test_worst_case_distance_for_empty_prediction():
    gt = np.zeros((30, 40), bool); gt[10:20, 10:20] = True
    m = evaluate(np.zeros_like(gt), gt)
    assert np.isnan(m["hd95"]) and m["hd95_wc"] == np.hypot(30, 40) == 50.0


def test_explain_agrees_with_metrics():
    from types import SimpleNamespace
    from esrg.explain import explain
    gt = np.zeros((40, 40), bool); gt[10:30, 10:30] = True
    pred = np.zeros_like(gt); pred[12:34, 8:28] = True
    seed = np.zeros_like(gt); seed[18:22, 18:22] = True
    st = SimpleNamespace(key="seed", image=seed, seconds=0.0, name="5 · Seed core")
    res = SimpleNamespace(mask=pred, gt=gt, stages=[st], meta={"seconds": 1.0, "seed_mode": "auto"},
                          stage=lambda k: st if k == "seed" else None)
    e, m = explain(res), evaluate(pred, gt)
    for k, mk in (("dsc", "dsc"), ("iou", "iou"), ("recall", "recall"), ("precision", "precision"),
                  ("hd95", "hd95"), ("assd", "assd"), ("leakage", "area_ratio")):
        assert abs(e[k]["value"] - m[mk]) < 1e-12, k
    assert e["seed"]["value"] == 1.0


def _square_case():
    img = np.full((60, 60), 40.0); img[20:40, 20:40] = 200.0
    img += np.random.RandomState(0).normal(0, 2, img.shape)
    L = log_transform(normalize(img))
    mask = np.ones((60, 60), bool)
    core = np.zeros((60, 60), bool); core[28:32, 28:32] = True
    return L, mask, core


def test_growth_recording_changes_nothing_and_obeys_eq_3_8_to_3_11():
    """record=True only logs: same region, same passes; every logged decision follows the rules."""
    L, mask, core = _square_case()
    cfg = Config()
    r0, t0 = grow_esrg(L, mask, core, 0.02, cfg)
    r1, t1 = grow_esrg(L, mask, core, 0.02, cfg, record=True)
    assert np.array_equal(r0, r1) and t0["passes"] == t1["passes"]
    bounds = {d["pass"]: d for d in t1["pass_detail"]}
    for p, r, c, Lv, ref, delta, dev, decision in t1["events"]:
        d = bounds[p]
        assert abs(delta - abs(Lv - ref)) < 1e-12 and abs(Lv - L[r, c]) < 1e-12
        assert abs(d["T_L"] - cfg.k_local * d["sigma"]) < 1e-12
        assert abs(d["sigma"] - max(d["s"], d["sigma_floor"])) < 1e-12
        if decision == "absorb":
            assert delta <= d["T_L"] and dev <= d["T_G"]
        elif decision == "reject":
            assert dev > d["T_G"]
        else:
            assert delta > d["T_L"]
    absorbed = sum(1 for e in t1["events"] if e[7] == "absorb")
    assert absorbed == r1.sum() - core.sum()


def _test_images():
    import glob
    return sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "segmentation_task", "test", "images", "*.jpg")))


def test_stage_report_covers_every_stage():
    """The telemetry has a report, titled first and with an OUTPUT, for every stage of a real run."""
    from esrg import run
    from esrg.stage_report import report
    files = _test_images()
    if not files:
        return                                   # dataset not present: nothing to run on
    for cfg in (Config(), Config(method="srg"),
                Config(method="srg", srg_use_log_local=True, srg_use_stopping=True)):
        res = run(files[0], cfg, record=True)
        for st in res.stages:
            kinds = [k for k, _ in report(res, cfg, st.key)]
            assert kinds[0] == "head" and "output" in kinds, (cfg.method, st.key)
            assert "flow" not in kinds and "settings" not in kinds
        assert any(k == "output" for k, _ in report(res, cfg, "evaluation", ["dsc"]))


def test_stage_report_reflects_the_runs_own_settings():
    """Adjusted hyperparameters and switched-off modules are what the report shows, and still check out."""
    from esrg import run
    from esrg.stage_report import report
    files = _test_images()
    if not files:
        return
    cfg = Config().replace(k_local=2.5, otsu_classes=4, rank_use_contrast=False, use_log_local=False)
    res = run(files[1], cfg, record=True)
    text, q_rows = [], []
    for st in res.stages:
        for kind, p in report(res, Config(), st.key):     # a default cfg must not override the run's
            if kind == "eq":
                text += p
            if kind == "table" and "Q from eq." in p["cols"]:
                q_rows += p["rows"]
    assert any("k_L = 2.5" in line for line in text), text
    assert any("ablation: global reference" in line for line in text)
    assert any("contrast term off" in line for line in text)
    assert q_rows and all(r[6] == r[7] for r in q_rows)
    for d in res.stage("growth").info["_trace"]["pass_detail"]:
        assert d["k_L"] == 2.5 and abs(d["T_L"] - 2.5 * d["sigma"]) < 1e-12


def _srg_case():
    img = np.full((50, 50), 60.0); img[15:35, 15:35] = 180.0
    img += np.random.RandomState(1).normal(0, 3, img.shape)
    img = normalize(img)
    mask = np.ones(img.shape, bool)
    core = np.zeros(img.shape, bool); core[23:27, 23:27] = True
    return img, mask, core


def test_srg_modules_off_is_the_published_algorithm():
    """With no module applied, region 1 follows the 1994 rule: unconditional, region-mean delta."""
    from esrg.growing import grow_srg, grow_srg_auto
    img, mask, core = _srg_case()
    region, trace = grow_srg_auto(img, mask, core, Config(), record=True)
    assert trace["stop_reason"].startswith("tessellation complete (no stopping")
    assert all(e[8] == "absorb" for e in trace["events"])
    assert len(trace["events"]) == region.sum() - core.sum()
    labels = core.astype(np.int32); labels[2, 2] = 2
    lab, t2 = grow_srg(img, labels, record=True)
    assert (lab > 0).all() and all(e[8] == "absorb" for e in t2["events"])


def test_srg_with_termination_obeys_its_bounds():
    """SRG + adaptive termination: every region-1 decision follows T_L and T_G, declined
    pixels go to the background, and the tumor never crosses the square's edge. (The
    background grid also seeds inside the square, so region 1 shares it, as in SRG.)"""
    from esrg.growing import grow_srg_auto
    img, mask, core = _srg_case()
    L = log_transform(img)
    stop = dict(k_L=1.5, k_G=3.0, sigma_floor=0.02)
    region, trace = grow_srg_auto(L, mask, core, Config(), record=True, local_radius=3, stop=stop)
    for r, c, g, ref, d, dev, t_l, t_g, verdict in trace["events"]:
        assert abs(d - abs(g - ref)) < 1e-12
        if verdict == "absorb":
            assert d <= t_l and dev <= t_g
        elif verdict == "decline >T_L":
            assert d > t_l
        else:
            assert verdict == "decline >T_G" and dev > t_g
    assert trace["declined"] == sum(1 for e in trace["events"] if e[8] != "absorb")
    truth = np.zeros(img.shape, bool); truth[15:35, 15:35] = True
    assert region.sum() > core.sum() and not (region & ~truth).any()
    assert trace["declined"] > 0


def test_srg_lists_only_the_stages_it_uses():
    """Plain SRG has no log stage; SRG manual also has no head-mask or candidate stage."""
    from esrg import run
    files = _test_images()
    if not files:
        return
    keys = lambda cfg, pts=None: [s.key for s in run(files[0], cfg, manual_points=pts).stages]
    assert "log" not in keys(Config(method="srg"))
    pts = [(200, 200, 1), (40, 40, 2)]
    k = keys(Config(method="srg", seed_mode="manual"), pts)
    assert not {"mask", "candidates", "log"} & set(k), k
    assert "log" in keys(Config(method="srg", seed_mode="manual", srg_use_stopping=True), pts)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
