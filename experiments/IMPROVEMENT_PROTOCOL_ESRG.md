# ESRG improvement protocol (BraTS DEVELOPMENT set only)

Pre-declared on 2026-10-13, **before any improvement run**. Referenced from §10 of
`PROTOCOL_BRATS.md`. The TEST set is not touched. This document is not edited in place;
changes go in §7.

## 1. Why
The DEV diagnostic (`outputs/diagnostic_esrg_dev/analysis.txt`) found that ESRG's glioma growth is
bimodal: 31% of planted-seed slices flood the brain (area > 10× the tumor, DSC 0.09) and 33% stop
under half the tumor. The stopping bounds T_L = k_L·σ_A and T_G = k_G·σ_A inherit the seed's spread
σ_A. When σ_A > 0.10, 56% of slices flood; when σ_A is at the noise floor (60% of meningiomas),
precision is high and recall low.

## 2. The two changes (both default OFF, so existing results are unchanged)
- **Change 1: bound calibration.** `sigma_cap`: σ_A = min( max(s_A, σ_floor), max(σ_cap, σ_floor) ).
  The noise floor `sigma_floor_min` is also re-tuned (it was fixed at 0.02).
- **Change 2: flood guard.** `flood_frac`: if the grown region exceeds this fraction of the brain
  mask, regrow with k_L and k_G multiplied by `flood_shrink` = 0.7 (cumulative, up to
  `flood_tries` = 3 times). `flood_frac` is set from the DEV ground-truth size prior (the 95th and
  99th percentile of tumor-core area ÷ brain-mask area on fold A), not from any method's output.

## 3. Equal effort for the baseline
Every change that exists for SRG + ASC is applied to it with the same search procedure and the
same selection rule. Plain 1994 SRG has no parameters and is unchanged.

## 4. Design
- **Folds:** DEV (518 patients) is split into folds A and B, stratified by tumor type (seed 2026,
  alternating after a seeded shuffle). Develop on A; confirm on B.
- **Seeds and metric:** planted P1 seeds (`experiments/evaluate.py: seed_protocol`);
  objective = (glioma mean DSC + meningioma mean DSC) / 2, as in tuning.
- **Round 1 (change 1).** ESRG: σ_cap ∈ {none, 0.04, 0.06, 0.08, 0.12} × (k_L, k_G) ∈
  {(2.0, 3.0), (2.5, 3.5), (3.0, 4.0)} × σ_floor ∈ {0.02, 0.03} = 30 configurations.
  SRG + ASC: σ_cap ∈ {none, 0.06, 0.12} × (k_L, k_G) ∈ {(3, 4), (5, 7), (6, 8)} ×
  σ_floor ∈ {0.02, 0.03} = 18 configurations.
- **Round 2 (change 2), on the Round-1 result.** ESRG: flood_frac ∈ {off, P95, P99} ×
  (k_L, k_G) ∈ {(2.0, 3.0), (2.5, 3.5), (3.0, 4.0), (4.0, 5.0)} = 12.
  SRG + ASC: flood_frac ∈ {off, P95, P99} × {(3, 4), (5, 7), (6, 8)} = 9.
- **Reference arms** on both folds: current ESRG (k 2.0/3.0, r 1), current SRG + ASC (k 5.0/7.0),
  plain SRG.

## 5. Selection and acceptance (fixed now)
1. Within a round, the best configuration on fold A wins. Ties within 0.005: fewer parameters
   changed, then the smaller change.
2. It is **accepted** only if, on fold B, its objective beats the current configuration by ≥ 0.02
   **and** neither tumor type's mean DSC falls by more than 0.01. Otherwise the round is rejected
   and the previous configuration is kept. There is no second try.
3. **Budget:** two rounds, ≤ 30 configurations each, no other changes. Any further idea needs
   a new entry in §7 before it is run.
4. **Stop rule:** after the rounds, if planted-seed ESRG remains more than 0.05 mean DSC below plain
   SRG on all of DEV, the improvement effort stops and the gap is reported.
5. The result is saved as a **new** file (`configs/esrg_brats_frozen_v2.json`). The earlier frozen
   file is not overwritten, and nothing is frozen for TEST without the user's approval.

## 6. Outputs
`outputs/improve_esrg/`: `folds.csv`, one results CSV per round and fold, `summary.json`, `run.log`.

## 7. Deviations log
- 2026-10-13 (bookkeeping, no effect on any result): the first run stopped after the Round-1 screening
  because the scoring step read the reference rows from the wrong file (they had already been computed
  in `refs.csv`). The lookup was fixed to read all result files and the run was resumed; no row was
  recomputed or changed.

- 2026-10-13 (exploratory side-test, requested by the user; **not** a selection round): automatic
  competing seeds (change 4) are tried in a separate script, `experiments/explore_auto_competitors.py`,
  with no change to `esrg/` and no change to the v2 parameters or to the stop-rule decision. Healthy
  tissue and background seeds are placed from the image alone (no ground truth): background = the air
  pixel farthest from the brain; healthy = the typical-intensity deep brain pixel in the hemisphere
  opposite the automatic tumor seed, >= 25 mm from it. Arms, all with the automatic tumor seed on DEV:
  ESRG v2 alone, ESRG v2 in compete mode with the automatic competitors, SRG with the same competitors;
  plus planted competitors in compete mode as an upper reference. Results are descriptive (mean DSC,
  paired Wilcoxon with effect size r); any adoption into ESRG would need its own pre-declared entry
  here and confirmation on data not used for this look. Output: `outputs/explore_auto_competitors/`.

  **Result of the side-test (exploratory, DEV, 518 slices; `outputs/explore_auto_competitors/analysis.txt`):**
  the automatic competitors landed correctly (healthy seed inside the tumor in 1%, background never) but
  did **not** help ESRG: automatic seed + ESRG v2 alone 0.546 mean DSC, with automatic competitors in
  compete mode 0.523 (paired r = -0.49); planted competitors also slightly hurt (0.663 -> 0.651). SRG
  with the same automatic competitors reached 0.599, the same as SRG with ground-truth-informed
  competitors (0.592; paired p = .47), and beat ESRG-compete by 0.080 (r = 0.55). Nothing was adopted.

## 8. Outcome (2026-10-13; DEV only, TEST untouched)
Both rounds ran as pre-declared. Decisions (gain measured on held-out fold B against the
configuration current at that point):

| Round | Arm | Winner on fold A | Gain on B | Decision |
|---|---|---|---|---|
| 1 | ESRG | σ_cap 0.08, k_L 3.0, k_G 4.0, floor 0.02 | +0.042 | **accepted** |
| 1 | SRG + ASC | its current setting (no cap) | +0.000 | rejected |
| 2 | ESRG | + flood guard at 0.15 of the brain, k_L 4.0, k_G 5.0 | +0.046 | **accepted** |
| 2 | SRG + ASC | flood guard at 0.27 of the brain | +0.008 | rejected |

Final ESRG: σ_cap 0.08, k_L 4.0, k_G 5.0, floor 0.02, flood guard 0.15 (shrink 0.7, up to 3 tries),
r = 1. Saved as `configs/esrg_brats_frozen_v2.json`; the earlier frozen file is unchanged and
nothing is frozen for TEST.

Mean DSC (average of glioma and meningioma), planted P1 seeds:

| Arm | Fold B (held out) | All DEV |
|---|---|---|
| ESRG v1 (earlier frozen) | 0.577 | 0.577 |
| **ESRG v2** | **0.666** | 0.663 |
| SRG + ASC | 0.716 | 0.718 |
| Plain SRG (3 planted seeds) | 0.734 | 0.740 |

**Stop rule (§5.4): triggered.** ESRG v2 stays 0.077 below plain SRG on all DEV (0.068 on fold B),
more than the 0.05 threshold, so the improvement effort stops here. The gap is reported.
Further changes (e.g. robust seed statistics, automatic competitors) need a new entry in §7 and,
because the stop rule was fixed in advance, the user's explicit decision to override it.
