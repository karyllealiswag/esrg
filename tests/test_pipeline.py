"""
test_pipeline.py — Smoke and correctness tests.

Purpose : Guard the metric math, normalization, growth stopping, and config rules.
Function : Small synthetic cases with known answers — perfect/disjoint DSC, output
          range, a bright square the grower must fill without leaking, seed-hit
          logic, rejection of an invalid k_global <= k_local, the seed-pixel
          report (per-region grouping, averages, click de-duplication), the
          evaluation pipeline (Cochran sample size, the
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
    g, m = cochran(966), cochran(759)         # glioma and meningioma pool of the BraTS 2023 study
    assert abs(g["n0"] - 384.1459) < 1e-3     # 1.959964^2 * 0.25 / 0.05^2
    assert g["n_min"] == 276                  # 384.146 / (1 + 383.146 / 966) = 275.1
    assert m["n_min"] == 256                  # 384.146 / (1 + 383.146 / 759) = 255.2
    assert abs(margin(300, 966) - 0.0470) < 5e-4 and abs(margin(300, 759) - 0.0439) < 5e-4


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
    for k, mk in (("dsc", "dsc"), ("recall", "recall"), ("precision", "precision"),
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


def _with_competitor(core):
    """Seed map: the tumor core as id 1 and one planted competitor (id 2) in the corner."""
    labels = core.astype(np.int32)
    labels[2:5, 2:5] = 2
    return labels


def test_srg_modules_off_is_the_published_algorithm():
    """With no module applied, region 1 follows the 1994 rule: unconditional, region-mean
    delta, and the tessellation allocates every pixel."""
    from esrg.growing import grow_srg
    img, mask, core = _srg_case()
    lab, trace = grow_srg(img, _with_competitor(core), record=True)
    assert trace["stop_reason"].startswith("tessellation complete (no stopping")
    assert (lab > 0).all() and all(e[8] == "absorb" for e in trace["events"])
    assert len(trace["events"]) == (lab == 1).sum() - core.sum()


def test_srg_with_termination_obeys_its_bounds():
    """SRG + adaptive termination: every region-1 decision follows T_L and T_G, declined
    pixels go to the competitor, and the tumor never crosses the square's edge."""
    from esrg.growing import grow_srg
    img, mask, core = _srg_case()
    L = log_transform(img)
    stop = dict(k_L=1.5, k_G=3.0, sigma_floor=0.02)
    lab, trace = grow_srg(L, _with_competitor(core), record=True, local_radius=3, stop=stop)
    region = lab == 1
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


def test_srg_alone_fills_the_image_and_asc_stops_it():
    """E7b: one seed and no stopping rule absorbs everything; ASC alone bounds the tumor."""
    from esrg.growing import grow_srg
    img, mask, core = _srg_case()
    lab, trace = grow_srg(img, core.astype(np.int32))
    assert (lab == 1).all() and "warning" in trace
    L = log_transform(img)
    lab, _ = grow_srg(L, core.astype(np.int32), stop=dict(k_L=1.5, k_G=3.0, sigma_floor=0.02))
    truth = np.zeros(img.shape, bool); truth[15:35, 15:35] = True
    assert (lab == 1).sum() > core.sum() and not ((lab == 1) & ~truth).any()


def test_esrg_compete_mode():
    """No competitor pixel = ESRG as proposed; with competitors and ASC off every pixel is
    allocated; with ASC on the tumor stays inside the square."""
    from esrg.growing import grow_esrg
    img, mask, core = _srg_case()
    L = log_transform(img)
    cfg = Config()
    plain, _ = grow_esrg(L, mask, core, 0.02, cfg)
    same, _ = grow_esrg(L, mask, core, 0.02, cfg, competitors=np.zeros(L.shape, np.int32))
    assert (plain == same).all()
    comp = _with_competitor(core); comp[comp == 1] = 0
    _, tr = grow_esrg(L, mask, core, 0.02, cfg.replace(use_stopping=False), competitors=comp)
    assert (tr["labels"] > 0).all() and tr["unlabeled"] == 0
    region, tr = grow_esrg(L, mask, core, 0.02, cfg, competitors=comp)
    truth = np.zeros(img.shape, bool); truth[15:35, 15:35] = True
    assert region.sum() > core.sum() and not (region & ~truth).any()
    # The competitor took the rest, except declined pixels the tumor encloses (unreachable,
    # as with SRG + ASC); post-processing fills those holes.
    left = tr["labels"] == 0
    assert tr["unlabeled"] == left.sum() and not (left & ~truth).any()


def _stripped_case(edge_tumor=False, radius=9):
    """A skull-stripped slice: zero outside a disk 'brain', a zero hole (CSF) inside it,
    and a bright tumor, either deep or touching the brain edge."""
    yy, xx = np.mgrid[:120, :120]
    brain = (yy - 60) ** 2 + (xx - 60) ** 2 <= 50 ** 2
    img = np.where(brain, 100.0, 0.0) + np.random.RandomState(2).normal(0, 2, brain.shape) * brain
    img[55:60, 55:60] = 0.0                            # dark CSF pocket inside the brain
    cy, cx = (60, 17) if edge_tumor else (60, 80)
    tumor = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2) & brain
    img[tumor] = 220.0
    return np.clip(img, 0, 255), brain, tumor


def test_pre_stripped_brain_mask_and_default_path():
    """pre_stripped: H is the non-zero brain with holes filled; the default path is untouched."""
    from esrg.preprocessing import head_mask
    img, brain, _ = _stripped_case()
    H, depth, info = head_mask(img, Config(pre_stripped=True))
    assert info.get("pre_stripped") and (H == brain).all()
    assert depth[60, 60] > depth[60, 15] and depth[~H].max() == 0
    _, _, info0 = head_mask(img, Config())
    assert "pre_stripped" not in info0 and "otsu_t" in info0["_steps"]


def test_min_clearance_decides_cortical_tumors():
    """A tumor touching the brain edge is rejected by the clearance filter at a positive
    value and kept at 0, which is why 0 is in the BraTS seeding grid."""
    from esrg.preprocessing import head_mask
    from esrg.seeds import select_seed
    img, _, tumor = _stripped_case(edge_tumor=True)
    for clearance, kept in ((8.0, False), (0.0, True)):
        cfg = Config(pre_stripped=True, min_clearance=clearance, interior_frac=0.0)
        H, depth, _ = head_mask(img, cfg)
        core, info = select_seed(img, H, depth, cfg)
        assert core.any() == kept, (clearance, info["components"])
        if kept:
            assert not (core & ~tumor).any()


def test_seed_protocol_obeys_its_rules():
    """Planted seeds: tumor inside the core, P2/P3 4-8 mm from P1, healthy seed in the
    other hemisphere and far from the core, background outside the brain."""
    from experiments.evaluate import seed_protocol, MM, HEALTHY_CLEAR, BG_CLEAR
    from scipy import ndimage as ndi
    # Brain about 360 px across, as on a 512-px BraTS slice; tumor right of the midline.
    yy, xx = np.mgrid[:512, :512]
    brain = (yy - 256) ** 2 + (xx - 256) ** 2 <= 180 ** 2
    tumor = (yy - 256) ** 2 + (xx - 340) ** 2 <= 30 ** 2
    img = np.where(brain, 100.0, 0.0)
    img[tumor] = 220.0
    seeds, log = seed_protocol(img, tumor, Config(), "synthetic.png")
    p1 = seeds[1][0]
    assert seeds[1][0][:2] == np.unravel_index(int(np.argmax(ndi.distance_transform_edt(tumor))), tumor.shape)
    for k in (2, 3):
        d = np.hypot(seeds[k][0][0] - p1[0], seeds[k][0][1] - p1[1]) / MM
        assert tumor[seeds[k][0][:2]] and 4 <= d <= 8, d
    away = ndi.distance_transform_edt(~tumor)
    for k in (1, 2, 3):
        (tr, tc, t1), (hr, hc, t2), (br, bc, t3) = seeds[k]
        assert (t1, t2, t3) == (1, 2, 3)
        assert brain[hr, hc] and away[hr, hc] >= HEALTHY_CLEAR
        assert not brain[br, bc] and ndi.distance_transform_edt(~brain)[br, bc] >= BG_CLEAR
    assert log["healthy_rule"] == "contralateral" and all(seeds[k][1][1] < 256 for k in (1, 2, 3))
    assert seed_protocol(img, tumor, Config(), "synthetic.png")[0] == seeds  # deterministic


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


def test_analysis_decision_rule_and_placement_sd():
    """An effect counts only with Holm p < .05 and |r| >= 0.1; placement SD is the n-1 SD over P1-P3."""
    import pandas as pd
    from experiments.analyze_brats import _call, placement_sd
    assert _call({"p_holm": 0.01, "r": 0.25}) == "better"
    assert _call({"p_holm": 0.01, "r": -0.25}) == "worse"
    assert _call({"p_holm": 0.01, "r": 0.05}) == "no effect"
    assert _call({"p_holm": 0.20, "r": 0.60}) == "no effect"
    df = pd.DataFrame({"file": ["a"] * 3 + ["b"] * 3, "tumor": "glioma",
                       "config": ["P_SRG", "P_SRG_2", "P_SRG_3"] * 2,
                       "dsc": [0.6, 0.7, 0.8, 0.5, 0.5, 0.5]})
    sd = placement_sd(df, "pooled")
    assert abs(sd["a"] - 0.1) < 1e-12 and sd["b"] == 0


def test_sigma_cap_and_flood_guard_defaults_off_and_behave():
    """Defaults leave growth unchanged; the cap clamps sigma_A (the floor always wins); the flood
    guard regrows a flooding region with a tighter bound."""
    from esrg.growing import grow_esrg, _sigma_A
    assert _sigma_A(0.30, 0.02) == 0.30 and _sigma_A(0.30, 0.02, 0.06) == 0.06
    assert _sigma_A(0.01, 0.02, 0.06) == 0.02 and _sigma_A(0.30, 0.05, 0.03) == 0.05
    img, mask, core = _srg_case()
    L = log_transform(img)
    base, tb = grow_esrg(L, mask, core, 0.02, Config())
    same, _ = grow_esrg(L, mask, core, 0.02, Config(sigma_cap=None, flood_frac=None))
    assert (base == same).all()
    tight, tt = grow_esrg(L, mask, core, 0.02, Config(sigma_cap=0.02))
    assert tight.sum() <= base.sum() and all(p["sigma"] <= 0.0201 for p in tt["passes"])
    # whole pipeline: a stopping bound loose enough to flood a flat noisy image, then guarded
    from esrg import run
    files = _test_images()
    if not files:
        return
    loose = Config(k_local=6.0, k_global=9.0)
    flooded = run(files[0], loose, manual_points=None)
    guarded = run(files[0], loose.replace(flood_frac=0.05, flood_shrink=0.5, flood_tries=4))
    g_info = guarded.stage("growth").info
    if flooded.stage("growth").image.sum() > 0.05 * flooded.stage("mask").image.sum():
        assert g_info.get("flood guard retries", 0) >= 1
        assert guarded.stage("growth").image.sum() < flooded.stage("growth").image.sum()
    off = run(files[0], loose)
    assert (off.mask == flooded.mask).all()


def test_v3_acceptance_rule_truth_table():
    """_esrg_decide: rule 'off' is the earlier order (local bound, then global); veto rejects
    healthy-near pixels; rescue absorbs a local failure that passes the global bound and is
    not healthy-near, and otherwise skips instead of ending the pass."""
    from esrg.growing import _esrg_decide as D
    # g, d, ref_mu, t_local, t_global, mu_H, rule, stopping
    assert D(5.0, 0.5, 5.0, 0.2, 1.0, None, "off", True) == ("stop", "stop")
    assert D(6.5, 0.1, 5.0, 0.2, 1.0, None, "off", True) == ("reject", "reject")
    assert D(5.1, 0.1, 5.0, 0.2, 1.0, None, "off", True) == ("absorb", "absorb")
    assert D(4.1, 0.1, 5.0, 0.2, 1.0, 4.0, "veto", True) == ("reject", "veto")      # nearer healthy 4.0
    assert D(4.1, 0.1, 5.0, 0.2, 1.0, 4.0, "veto", False) == ("reject", "veto")     # veto without ASC
    assert D(5.6, 0.5, 5.0, 0.2, 1.0, 4.0, "rescue", True) == ("absorb", "rescue")
    assert D(4.3, 0.5, 5.0, 0.2, 1.0, 4.0, "rescue", True) == ("skip", "skip")      # healthy-near
    assert D(6.5, 0.5, 5.0, 0.2, 1.0, 4.0, "rescue", True) == ("skip", "skip")      # beyond T_G
    assert D(5.6, 0.5, 5.0, 0.2, 1.0, 4.0, "both", True) == ("absorb", "rescue")
    assert D(4.3, 0.1, 5.0, 0.2, 1.0, 4.0, "both", True) == ("reject", "veto")


def test_v3_healthy_prototype_estimators():
    """median / mode / otsu3 all find the dominant healthy level of a brain with a bright lesion."""
    from esrg.preprocessing import healthy_prototype
    rng = np.random.RandomState(3)
    g = np.concatenate([rng.normal(4.5, 0.05, 9000), rng.normal(3.5, 0.05, 500), rng.normal(5.6, 0.05, 500)])
    G = g.reshape(100, 100); region = np.ones_like(G, bool)
    for how in ("median", "mode", "otsu3"):
        assert abs(healthy_prototype(G, region, how) - 4.5) < 0.1, how


def test_v3_srg_mirror_uses_the_same_decision():
    """SRG's region 1 with the healthy rule declines exactly the pixels _esrg_decide rejects."""
    from esrg.growing import _Region1, _esrg_decide
    img = np.array([[5.0, 5.1, 4.1, 5.6]]); core = np.array([[True, False, False, False]])
    r1 = _Region1(img, core, None, dict(k_L=1.5, k_G=3.0, sigma_floor=0.2), dict(mu=4.0, rule="both"))
    for c, d in ((1, 0.1), (2, 0.9), (3, 0.6)):
        act, _ = _esrg_decide(img[0, c], d, r1.stats.mean, 1.5 * r1.sigma(), 3.0 * r1.sigma(), 4.0, "both", True)
        assert (r1.decide(0, c, d) == "absorb") == (act == "absorb")


def test_v3_srg_connectivity_option():
    """8-connected SRG (as published) and the 4-connected reproduction both tessellate fully."""
    from esrg.growing import grow_srg
    img, mask, core = _srg_case()
    seeds = _with_competitor(core)
    for k in (4, 8):
        lab, _ = grow_srg(img, seeds, connectivity=k)
        assert (lab > 0).all()


def test_brats_arms_mirror_esrg_settings():
    """Final BraTS evaluation: SRG mirrors read ESRG's settings (no separate SRG tuning), the ASC-only
    arms omit the log-unit sigma cap, SRG is 8-connected, TEST needs explicit approval."""
    import experiments.evaluate as E
    if not os.path.isfile(E.FROZEN):
        return
    b = E.base_config()
    cfg = {k: b.replace(seed_mode=v[2], **v[1]) for k, v in E.CONFIGS.items()}
    assert b.global_running and b.healthy_rule == "off" and b.srg_k_local is None and b.srg_local_radius is None
    assert cfg["P_SRG"].method == "srg" and cfg["P_SRG"].srg_connectivity == 8
    assert "P_SRG_S" not in cfg and "P_SRG_LS" not in cfg and len(cfg) == 12
    for k in ("P_SRG_L", "P_SRG_solo_S"):
        assert cfg[k].srg_k("local") == b.k_local and cfg[k].srg_k("global") == b.k_global
        assert cfg[k].srg_radius() == b.local_radius
    assert cfg["P_SRG_solo_S"].sigma_cap is None and cfg["P_SRG_solo_S"].srg_use_stopping
    assert cfg["P_ESRG_noL"].sigma_cap is None and not cfg["P_ESRG_noL"].use_log_local
    assert cfg["P_SRG_solo_S"].flood_frac == b.flood_frac and E.CONFIGS["P_SRG_solo_S"][4] == (1,)
    assert cfg["P_ESRG_C_noS"].esrg_compete and not cfg["P_ESRG_C_noS"].use_stopping


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
