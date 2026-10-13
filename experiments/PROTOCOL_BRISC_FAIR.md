# Evaluation Protocol: Fair ESRG vs. SRG on BRISC 2025 (25% rerun)

Written on 2026-10-13, **before any run of this evaluation**. Once fixed, this document is not
edited in place; any change goes in §9 (Deviations log) with its date and reason.

The BraTS 2023 study (`experiments/PROTOCOL_BRATS.md`) is on hold and is not affected by this
protocol.

## 0. Why this rerun
The original BRISC evaluation (Chapter 4; `outputs/evaluation/`, archived evaluator
`experiments/evaluate_brisc.py`) did not give both methods the same inputs:
- the SRG baseline grew against a grid of dozens of background seeds, so the tumor region was
  boxed in (planted SRG: DSC 0.148, recall 0.09, precision 0.998);
- ESRG's planted seed was eroded ("purified") while SRG's was not;
- no arm isolated a single method.

This rerun gives both methods identical, realistic seeds, and isolates each proposed method in
its own experiment.

## 1. Data and sample
- **Frame:** the existing 954-slice BRISC sample (`outputs/evaluation/sample.csv`). It has 9 strata
  (glioma / meningioma / pituitary × axial / coronal / sagittal) of 106 slices each.
- **Subsample: 27 slices per stratum = 243 slices (25.5%).** Within each stratum, the
  already-selected slices are taken in their recorded random `draw_order` (seed 2026,
  `outputs/evaluation/frame.csv`), and the first 27 are kept. The subsample is nested in the
  current sample and fully reproducible.
- Ground truth: BRISC tumor masks (`find_mask_for`).
- **Parameters:** the BRISC-frozen defaults in `esrg/config.py`, **not re-tuned**.
  - The SRG modules use the same k_L, k_G and r as ESRG.
  - Seed purification is removed (both methods start from the identical click disk).

## 2. Seed protocol (pixels; BRISC has no pixel spacing)
- **Determinism:** seeds are fixed by `crc32(file|placement|type)` and logged to `manual_seeds.csv`.
- **Who uses which seeds:**
  - Every arm of a comparison receives the same seeds.
  - The ground truth places the planted seeds only, to simulate a careful operator; the automatic tumor seed never sees it.
  - **One seed per type** at each placement P1 (reference), P2 and P3. The tumor disk has radius 3 px.

| Type | Pool | P1 | P2 / P3 |
|---|---|---|---|
| 1 Tumor | GT pixels ≥ 3 px from the GT edge (fallback: any GT pixel) | Deepest GT pixel | Random pool pixel 8–16 px from P1; P3 also ≥ 8 px from P2 |
| 2 Healthy tissue | Head mask, depth > 0.4·max depth, ≥ 12 px from the GT | Random | 8–16 px from P1's seed (P3 also ≥ 8 px from P2's) |
| 3 Background (air) | Outside the head mask, ≥ 8 px from it | Random | as type 2 |
| 4 Scalp/skull | Head mask, ≥ 12 px from the GT, depth 2–8 px | Random | as type 2 |

- The pools are those of the original BRISC manual protocol (`manual_protocol_points`), reduced to one seed each.
- If no pool pixel lies in the 8–16 px ring, the pool pixel whose distance to P1 is closest to 12 px is used, and the actual offset is recorded.
- How each arm reads the seeds:
  - SRG uses all four types.
  - A_SRG uses the automatic tumor seed with P1's types 2–4.
  - ESRG in its normal mode reads only type 1; in compete mode (E8) it also uses types 2–4.

## 3. Arms and experiments
Notation: A_ = automatic tumor seed; P_ = planted; +L / −L and +S / −S = LLDM and ASC on / off;
_C = ESRG compete mode.

| Exp | Baseline → Enhanced | Objective | Primary metric |
|---|---|---|---|
| E1 | P_SRG → A_ESRG | Overall (system) | DSC |
| E2 | P_SRG → P_ESRG | Overall (algorithm) | DSC |
| E3 | A_ESRG | 1 | SHR |
| E4a | A_ESRG vs P_SRG{P1, P2, P3} | 1 | DSC per condition; seed-placement SD |
| E4b | A_SRG vs P_SRG{P1, P2, P3} | 1 | as E4a |
| E5 | P_SRG → P_SRG+L | 2 | Recall (DSC no-harm) |
| E6 | P_ESRG(−L, +S) → P_ESRG(+L, +S) | 2 | Recall (DSC no-harm) |
| E7a | P_SRG → P_SRG+S (with competitors) | 3 | Precision, HD95 (DSC no-harm) |
| E7b | P_SRG → P_SRG+S (tumor seed only) | 3 | Precision, HD95 (DSC descriptive; degenerate baseline by design) |
| E8 | P_ESRG_C(+L, −S) → P_ESRG_C(+L, +S) | 3 | Precision, HD95 (DSC no-harm) |

**Determinism:** A_ESRG and A_SRG are run twice more in fresh processes; their masks (`mask_sha1`)
must be identical.

## 4. Metrics
- **TP / FP / FN** come from the final mask and the ground truth (`esrg.metrics.confusion`).
- **DSC** = 2TP / (2TP + FP + FN).
- **Recall** = TP / (TP + FN), for undersegmentation.
- **Precision** = TP / (TP + FP), for leakage.
- **HD95** (px): the 95th percentile of the pooled boundary distances; the image diagonal for an empty prediction.
- **SHR**: the share of slices whose whole seed core lies inside the ground truth; "no candidate" counts as a miss.
- **Seed-placement SD**: the SD of DSC over P1–P3 per slice; 0 by construction for automatic seeding.

## 5. Statistics
- **Unit:** one slice.
- **Reporting:** results are reported per tumor type and pooled. Plane and BRISC split (train/test) are descriptive.
- **Procedures:**
  - descriptives: median [IQR], mean ± SD, median paired difference;
  - Shapiro–Wilk on the paired differences, as justification only;
  - Wilcoxon signed-rank (two-sided, α = .05) with r = Z/√N;
  - Holm correction within each experiment's family of tested metrics;
  - Wilson 95% CI for SHR;
  - E4: Friedman + Kendall's W over {auto, P1, P2, P3}, then post-hoc Wilcoxon (auto vs each placement, Holm).
- **Power:** smallest detectable standardised paired effect (80% power): d_z ≈ 0.18 pooled (n = 243) and ≈ 0.31 per tumor type (n = 81).

## 6. Decision rules (fixed now)
An effect counts only if Holm p < .05 **and** r ≥ 0.1.
- **Overall:** ESRG DSC significantly higher than P_SRG in E1 and in E2.
- **Objective 1:**
  - (a) SHR Wilson lower bound ≥ 70%, **judged per tumor type**; the pooled SHR is reported, not judged.
  - (b) Automatic seed-placement SD = 0 while the manual median SD > 0.
  - (c) Auto DSC not significantly lower than any placement.
- **Objective 2:** recall significantly higher and DSC not significantly lower, in both E5 and E6.
- **Objective 3:** precision significantly higher, HD95 significantly lower and DSC not significantly lower, in both E7a and E8. E7b is supporting evidence.

All results are reported regardless of direction.

## 7. Limitations (stated before the run)
- **Tuning overlap.** 791 of the 954 sampled slices come from the BRISC train split on which the
  ESRG parameters were tuned. This favours ESRG, because SRG has no tuned parameters. A per-split
  table is reported.
- **Unknown patient identity.** BRISC slices may share patients, so independence is assumed, not known.
- **Distances in pixels.** BRISC has no pixel spacing.
- **Head mask beyond the brain.** On coronal and sagittal slices the head mask includes the face
  and neck, so the healthy-tissue seed may land on non-brain tissue (as in the original protocol).
- **Known weak seeding.** In the original run, automatic seeding hit 15% of gliomas and 42% of
  pituitary tumors, so criterion 1(a) is not expected to be met for those types.

## 8. Outputs
- `outputs/evaluation_fair/`: `sample_25.csv`, `raw_results.csv`, `manual_seeds.csv`, `raw_results_repeat{k}.csv`, `results_fair.json` and the tables.
- **Code:** `experiments/evaluate_fair.py` and `experiments/analyze_fair.py`.

## 9. Deviations log
- 2026-10-13 (analysis): §6 rule 1(c) did not say which E4 sub-experiment decides it. It was
  applied to E4a (A_ESRG, the proposed system), and E4b is reported alongside. The verdict is
  the same for both (not met in every scope), so the choice does not change any conclusion.
