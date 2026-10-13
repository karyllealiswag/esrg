# Evaluation Protocol: ESRG vs. SRG on BraTS 2023 (GLI + MEN)

Version 2. **Checkpoint 1 decisions recorded** (§11). Once approved, this document is frozen.
Any later deviation goes in §10 (Deviations log) with its date and reason, and is never edited in place.

---

## 0. Data (the sampling frame)

| Item | Glioma (GLI) | Meningioma (MEN) |
|---|---|---|
| Source (outside repo, never modified) | `C:\Users\karyl\Downloads\brats_pool\brats_pool\glioma\` | `…\brats_pool\meningioma\` |
| Patients = scans = slices | **966** | **759** |
| Slices per patient | exactly 1 | exactly 1 |
| Image | 512 × 512 8-bit PNG, axial T1c, skull-stripped | same |
| Masks | `masks/` tumor core, `masks_enhancing/` enhancing; values {0, 255} | same |
| Enhancing ÷ core area (median) | 0.76 | 1.00 |
| Core area, native grid, median [IQR] (px) | 651 [340–1191] | 448 [250–895] |

- **Slice per patient:** each patient's slice is a **random eligible slice** (§2), produced upstream
  by the BraTS 2023 export (labels: core = 1 + 3, enhancing = 3).
- **Resolution:** BraTS is 240 × 240 at 1 mm, upsampled to 512 px, so **1 px = 240/512 = 0.469 mm**.
  HD95 is reported in mm. Pixel parameters keep the 512-px scale already used by `esrg/config.py`.
- **No edema masks:** whole-tumor masks are not available, which affects healthy-seed placement (§6.2).
- **Inference:** the frame is 1,725 BraTS 2023 patients (~77% of GLI's ~1,250 and ~76% of MEN's
  ~1,000). Results generalise to BraTS-type data, with the margins in §3.3.

The earlier exports (`brats2023/`, `bratsmen/`: 96 + 81 patients) are superseded and were
already removed from the repo by the user. They are not used.

## 1. Target mask

**Primary: tumor core** (labels 1 + 3), the standard BraTS "TC" region.
- It is what a T1c method can be expected to delineate: the enhancing tissue is bright, and the
  necrotic centre is recovered by hole-filling in post-processing.

**Secondary, descriptive only, glioma only: enhancing tumor** (label 3).
- In MEN the enhancing region *is* the core (median ratio 1.00), so a second MEN score would be
  redundant. In GLI about 24% of the core (median) is non-enhancing and near-isointense on T1c.
- Reporting GLI enhancing-only DSC separates "non-enhancing tissue missed" from "algorithm failure".
- No tests are run on it and no objective is judged on it.

## 2. Eligible slice

Axial T1c with **tumor-core area ≥ 150 px on the native 240-px grid** (≈ 150 mm², a disk about 14 mm across).
The manifests confirm a minimum of 150.

**This excludes small tumors and the thin top and bottom sections of every tumor.** No claim is
made about small lesions.

## 3. Patient-level split

### 3.1 Rules
- Unit = patient = (tumor type, 5-digit ID). GLI and MEN number patients independently, so
  GLI-00019 and MEN-00019 are different people. One scan per patient: the lowest session
  suffix. No duplicates exist in the pool, and the code asserts this.
- Seed **2026**, stratified by tumor type. Within each type, patients are sorted by ID, shuffled
  with `random.Random(2026)`, and the first n_test go to TEST; the rest go to DEVELOPMENT.
- No patient appears in both sets (asserted in code, verified in Stage 2).

### 3.2 Sample size
**Cochran minimum** (1977; 95% confidence, e = 0.05, p = 0.5, FPC):
- n₀ = 1.959964² × 0.25 / 0.05² = **384.15**
- GLI (N = 966): 384.15 / (1 + 383.15/966) = 384.15 / 1.3966 = 275.1 → **276**
- MEN (N = 759): 384.15 / (1 + 383.15/759) = 384.15 / 1.5048 = 255.3 → **256**

**Allocation decided at Checkpoint 1: 70% TEST / 30% DEVELOPMENT of the whole pool, per type
(proportional).** It exceeds the Cochran minimum in both strata.

| | Pool N | Cochran min | **TEST** (0.70 N) | **DEVELOPMENT** |
|---|---|---|---|---|
| GLI | 966 | 276 | **676** | **290** |
| MEN | 759 | 256 | **531** | **228** |
| Total | 1,725 | — | **1,207** | **518** |

Every pool patient is used, so no pool slice is unused.

### 3.3 Achieved precision and power
Margin of error of a proportion (e.g. SHR):

| | Within the pool (FPC) | For BraTS-type data (no FPC) |
|---|---|---|
| GLI, n = 676 | ± 2.1% | ± 3.8% |
| MEN, n = 531 | ± 2.3% | ± 4.3% |
| Pooled, n = 1,207 | ± 1.5% | ± 2.8% |

Smallest detectable standardised paired effect (two-sided α = .05, power 0.80, Wilcoxon ARE 0.955),
from d_z = 2.80 / √(0.955·n):
- GLI ≈ **0.11**
- MEN ≈ **0.12**
- pooled ≈ **0.08**

Consequence: tiny differences will reach p < .05. **Conclusions are therefore based on the effect
size r and the median paired difference**, with p-values required but not sufficient (§8).

## 4. Slice selection
- **TEST:** the patient's single pool slice, a seeded random eligible slice (decided at Checkpoint 1).
  A random eligible slice is used rather than the largest-core slice so that the easiest,
  best-contrasted section is not systematically chosen.
- **DEVELOPMENT:** the patient's single pool slice. Five slices per patient are not possible
  because the pool holds one; this is stated as a minor deviation from the brief.
- Tuning still averages per patient, and with one slice each, patient = slice.

## 5. Tuning procedure (DEVELOPMENT only)

### 5.1 Code changes that must precede tuning (Stage 2)
- `Config.pre_stripped` (§7.1).
- Seed purification removed.
- The 3-type planted seed protocol (§6.2).
- SRG automatic mode = automatic tumor seed + planted competitors (the background grid is removed).
- ESRG compete mode (used only in E8).

### 5.2 Growth objective
1. DSC against the tumor core, per patient. A failed or empty prediction scores DSC = 0.
2. Mean over patients within each type.
3. **Objective = (GLI mean + MEN mean) / 2.**

The median is logged but not used for selection.

### 5.3 Stages, grids and budget

**Stage T1: growth parameters**, using the planted P1 seeds (§6.2) so seeding error is excluded.

| Arm tuned | Grid | Configs |
|---|---|---|
| ESRG (P_ESRG) | k_local ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}; k_global ∈ {2.0, 2.5, 3.0, 4.0, 5.0}, k_global > k_local; local_radius ∈ {1, …, 5} | 29 × 5 = **145** |
| SRG + LLDM (P_SRG+L) | local_radius ∈ {1, …, 5} | **5** |
| SRG + ASC (P_SRG+S, with competitors) | the same 29 (k_local, k_global) pairs | **29** |
| SRG (P_SRG, 1994) | none: the published algorithm has no growth parameter | 1 |

- SRG receives the **identical grid values** for every parameter its variant reads, on the same
  slices and seeds.
- Fixed, not tuned: max_passes 4, log_eps 1.0, sigma_floor_min 0.02, post_open_radius 1,
  manual_seed_radius 3.
- Ablation arms (ESRG −LLDM in E6; ESRG compete ±ASC in E8; SRG+S without competitors in E7b)
  **reuse the frozen values of their parent arm** and are not re-tuned.

**Stage T2: automatic-seeding parameters**, shared by A_ESRG and A_SRG.
- **Objective: seed hit rate** on DEVELOPMENT, averaged per type and then across the two types
  (decided at Checkpoint 1). Being method-neutral, it favours neither growth rule.

| Parameter | Grid |
|---|---|
| otsu_classes | {3, 4} |
| seed_core_frac | {0.5, 0.7, 0.85} |
| min_cand_area | {20, 40, 80} |
| interior_frac | {0, 0.06, 0.12} |
| min_clearance | {0, 4, 8}; 0 is allowed because cortical tumors touch the brain outline |
| rank_w_contrast | {0.25, 0.5, 1.0} |
| rank_w_shape | {0, 0.2} |
| rank_w_radius | fixed at 1.0 (scale reference) |

That is **972 configurations**. Seed selection is evaluated without growth.

**Budget.** Measured cost: ~1.7 s per ESRG slice and ~2.7 s per SRG slice per core; 10 worker processes.
- T1 ≈ 145 × 518 × 1.7 / 10 ≈ 3.5 h (ESRG) plus ≈ 1.1 h (SRG variants).
- T2 is estimated at ≤ 6 h.
- **Hard cap: 12 h of wall time per stage.** If the timed estimate exceeds the cap, a seeded (2026)
  random subset of 300 grid points is used instead, and this is logged.

**Tie-breaks** (applied in order when objectives are within 0.005 for T1, or 0.5 percentage points for T2):
1. Fewer parameters changed from the BRISC-frozen defaults.
2. Smaller normalised distance from those defaults.
3. Lower k_local (T1) or higher min_clearance (T2), i.e. the more conservative choice.

**Logging.** Every configuration and its scores (objective, per-type mean and median, SHR, runtime)
go to `outputs/tuning_brats/`. The winner is frozen to `configs/esrg_brats_frozen.json`; the BRISC
defaults stay untouched. A sensitivity table shows the objective against each parameter, with the
others held at the winner.

## 6. Experimental arms (Stage 4, one TEST run)

### 6.1 Arms
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
| E7b | P_SRG → P_SRG+S (tumor seed only) | 3 | Precision, HD95 (DSC descriptive) |
| E8 | P_ESRG_C(+L, −S) → P_ESRG_C(+L, +S) | 3 | Precision, HD95 (DSC no-harm) |

Determinism check: A_ESRG and A_SRG are re-run twice in fresh processes, and the masks must be
identical (reported in one sentence).

### 6.2 Seed protocol
- **Positions are deterministic:** `crc32(file|placement)`, saved to `manual_seeds.csv`.
- **Distances are in mm** (1 mm ≈ 2.13 px). The seed disk has radius 3 px.
- **Tumor (type 1):** chosen from core pixels whose distance to the core edge is ≥ 3 px. P1 = the
  deepest core pixel. P2 = a random eligible pixel 4–8 mm from P1. P3 = another, 4–8 mm from P1
  and ≥ 4 mm from P2. Small tumors fall back to the nearest eligible pixel, and the actual
  distance is recorded.
- **Healthy tissue (type 2):** placed in the **hemisphere opposite the core centroid** (the midline
  is the centre column of the brain mask's bounding box), in brain pixels with depth > 0.4·max and
  ≥ 25 mm from the core.
  - Edema masks are unavailable, and edema is predominantly ipsilateral, so the contralateral side
    is the natural "normal tissue" choice.
  - If no pixel qualifies (e.g. a midline tumor), it falls back to the farthest eligible brain
    pixel from the core, and this is logged.
  - P2/P3: 4–8 mm from P1's healthy seed.
- **Background (type 3):** image = 0, ≥ 8 px outside the brain mask; P2/P3 as above.
- **Who uses which seeds:**
  - Every P_ arm uses P1.
  - A_SRG uses P1's types 2–3 with its automatic tumor seed.
  - ESRG in normal mode reads only type 1.
  - The ground truth is used only to simulate a careful operator; the automatic tumor seed never sees it.

## 7. Pipeline adaptations

### 7.1 `Config.pre_stripped` (default False, so the BRISC path is unchanged)
When True:
- the head mask = the non-zero brain mask, `fill_holes(largest_cc(img > 0))`, skipping Otsu,
  opening and the `ss_*` granulometry;
- depth = the EDT from the brain outline.

Interior and clearance filters then measure from the brain edge. Cortical tumors touch that edge,
which is why 0 is in the T2 grid for min_clearance and interior_frac. Stage 2 counts DEV cortical
tumors (core within 3 px of the brain edge) that are rejected under each value.

### 7.2 Dataset layout (`brats_eval/`, gitignored)
- `test/images`, `test/masks` and `dev/images`, `dev/masks`.
- File stems are identical between images and masks, and the `glioma_` / `meningioma_` prefix is kept.
- `split.csv` columns: patient, tumor, set, slice, file, core_area, enhancing_area. The tumor type
  comes from here, and the BRISC regex keeps working.
- Ground-truth outline overlays for the Stage 2 spot check go to `outputs/brats_spotcheck/`.

## 8. Metrics and statistics

**Metrics:**
- DSC, recall and precision against the tumor core.
- HD95 in mm (px × 0.469), with worst case = image diagonal for empty predictions.
- SHR: hit = the whole seed core inside the core mask; "no candidate" counts as a miss.
- Seed-placement SD: SD of DSC over P1–P3 per patient.
- Secondary: GLI enhancing-only DSC (descriptive).

**Unit of analysis.** One slice per TEST patient, so patient level = slice level. Results are
reported **per tumor type and pooled**. The pooled figure is the plain mean over the 1,207 patients
(56% GLI, 44% MEN, matching the pool's proportions).

**Statistics** (`experiments/stats.py`):

| Use | Procedure |
|---|---|
| Descriptives | median [IQR], mean ± SD, median paired difference |
| Choice of test | Shapiro–Wilk on paired differences (justification only) |
| **Main test** | **Wilcoxon signed-rank** (two-sided, α = .05) with r = Z/√N |
| Multiplicity | **Holm** within each experiment's metric family; pooled and each type are separate families |
| Proportions | **Wilson** 95% CI for SHR |
| E4 | Friedman + Kendall's W, then post-hoc Wilcoxon (auto vs each placement, Holm) |

**Decision rules (fixed):**
- **Effects count only if Holm p < .05 and r ≥ 0.1.**
- *Overall:* ESRG DSC significantly higher.
- *Objective 1:* **SHR's Wilson lower bound ≥ 70%**; auto seed-placement SD = 0 while the manual
  median SD > 0; auto DSC not significantly lower than any placement.
- *Objective 2:* recall significantly higher and DSC not significantly lower in **both** E5 and E6.
- *Objective 3:* precision significantly higher, HD95 significantly lower, and DSC not
  significantly lower in **both** E7a and E8. E7b is supporting evidence; its baseline is
  degenerate by design.

All results are reported regardless of direction, including poor glioma results.

## 9. Integrity rules
- Nothing is run on TEST before Checkpoint 3.
- TEST is run once.
- No commits without approval.
- `python -P` is used on data.
- The source pool is never modified or deleted.

**Disclosure.** Before this protocol existed, three slices (52, 55, 58) of GLI patient 00019 from
the superseded export were run once with the BRISC defaults to measure runtime (DSC 0 in all).
The pool holds a different slice of that patient (slice 68). No parameter was chosen from this,
and the seeded split is left untouched.

## 10. Deviations log
- Version 2: DEVELOPMENT has 1 slice per patient, not up to 5 (the pool has one per patient).
- Stage 2: the patient key is (tumor type, ID), not the ID alone. The first verification run
  counted the same number in both cohorts as one patient; the check was corrected, and the
  split itself was unaffected (it is drawn within each type).
- Stage 2: the noise floor σ_floor is now measured inside the brain/head mask H for **every**
  arm. SRG under manual seeding previously measured it over the whole image, where the zero
  background drives the MAD to 0. This affects only SRG + ASC, which now reads the same
  tissue-noise estimate as ESRG.
- Stage 2: under ESRG compete mode and SRG + ASC, a pixel declined by the tumor but enclosed
  by it cannot be reached by the 4-connected competitors and stays unlabeled. Post-processing
  hole-filling assigns it to the tumor. This is documented behaviour, not an error.

- Checkpoint 3 (2026-10-13, decided by the user on DEV results): the SRG + ASC winner
  (k_L = 3.0, the grid maximum) and the SRG + LLDM winner (r = 5, the grid maximum) lay at the
  top edge of the pre-specified grid and were still improving. Their grids were extended upward
  on DEV only: SRG + ASC adds (3.0, k_G ∈ {6, 7, 8}) and k_L ∈ {3.5, 4, 5, 6} × k_G ∈ {4, 5, 6, 7, 8}
  with k_G > k_L (17 pairs); SRG + LLDM adds r ∈ {6, 7, 8, 10}. ESRG's grid is unchanged (its
  winner is interior, or at r = 1, the smallest possible window). Same objective and tie-breaks.
- Checkpoint 3: the planted P1 rule (deepest core pixel) is **kept**, although on DEV it lies
  outside the enhancing tissue in 54% of gliomas (ESRG DSC 0.30 there vs 0.50 inside). Changing
  it after seeing which method it favours was declined; the effect is reported as a finding.
- Checkpoint 3 diagnostic (no change): raising max_passes from 4 to 10 did not improve ESRG
  DSC on 80 DEV slices (MEN 0.808 → 0.809, GLI 0.390 → 0.367), so P_max stays fixed at 4.

- 2026-10-13: a DEV-only ESRG improvement search (sigma_cap, flood guard) was pre-declared and run;
  see `IMPROVEMENT_PROTOCOL_ESRG.md`. Candidate parameters are in `configs/esrg_brats_frozen_v2.json`
  and are **not approved**. The Checkpoint 3 approval of frozen parameters is therefore still pending.

- 2026-10-13: at the user's request the earlier improvement stop rule is overridden by a new
  pre-declared protocol, `IMPROVEMENT_PROTOCOL_ESRG_V3.md` (running global mean, healthy-tissue
  rule, SRG made 8-connected as published; fresh DEV split D1/D2/D3). TEST remains untouched.

- 2026-10-13: final ESRG chosen by the user = v2 + running global mean (`configs/esrg_brats_frozen_v3.json`),
  adopted despite the v3 acceptance rule (see `IMPROVEMENT_PROTOCOL_ESRG_V3.md` §11). The
  evaluation pipeline is rehearsed on DEV first; the single TEST run waits for the user's go.

- 2026-10-13 (before the TEST run; decided by the user): **arms and TEST sample size.**
  - Arms P_SRG_S and P_SRG_LS are removed; **E5a is redefined as P_SRG (3 planted seeds) vs
    P_SRG_solo_S (tumor seed only + ASC)**, i.e. whether ASC can replace SRG's competing seeds; E7 is
    dropped. Objective 3 uses E5a and E6. 12 arms remain.
  - **TEST sample = 300 patients per tumor type (600).** Cochran (1977) for a proportion, z = 1.96,
    p = 0.5, e = 0.05: n0 = 384.15; finite-population correction on each stratum's pool
    (GLI N = 966, MEN N = 759) gives 276 and 256. Equal allocation at the larger requirement,
    inflated by about 9% as the allowance for non-evaluable cases, gives n_h = 300.
    Achieved 95% margins: GLI ±4.7, MEN ±4.4, pooled ±3.2 percentage points. Detectable paired
    effect (Wilcoxon, two-sided α = .05, power .80): d_z ≈ 0.165 per type, 0.117 pooled.
  - Draw: within each type's TEST list, files sorted, `random.Random(2026).shuffle`, first 300;
    the list is written to `outputs/evaluation_brats/test/test_sample.csv` before any slice runs
    (`evaluate.py --sample-per-type 300`). Pooled estimates weight the two types equally.
  - Single TEST run with the frozen v3 configuration, followed by the two determinism repeats.

- 2026-10-13 (TEST run complete): 600 patients x 12 arms = 7,200 rows, 0 errors; determinism
  600/600 identical for A_ESRG and A_SRG in both repeats. Part of the main run ran on battery power
  (CPU throttled; slices ~375-600 took ~3x longer). Only the `seconds`/timing columns are affected;
  timings are reported as indicative. Results: outputs/evaluation_brats/test/ (summary.md, results.json).

- 2026-10-13 (after the TEST run; user decision): a **secondary, descriptive** comparison S1 = P_SRG_solo vs
  P_ESRG (equal operator effort: one tumor click) was added to `analyze_brats.py` (`SECONDARY`). It uses rows
  already collected, is not part of the pre-specified verdicts, and is labelled as post hoc in Chapters 3-4.
  No TEST row was changed, added, or re-run; all strict verdicts are reported unchanged.

## 11. Checkpoint 1 decisions
| Decision | Choice |
|---|---|
| Data | `brats_pool` (966 GLI + 759 MEN, one slice each) replaces the 96 + 81 export |
| Allocation | 70 / 30 of the whole pool, per type |
| Pool slice rule | Random eligible slice |
| Seeding tuning objective | Seed hit rate (shared by A_ESRG and A_SRG) |
| SHR target | Wilson lower bound ≥ 70% |
| Healthy seed | Contralateral hemisphere (proposed default; no objection raised) |
