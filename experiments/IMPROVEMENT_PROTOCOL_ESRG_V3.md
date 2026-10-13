# ESRG improvement protocol v3 (BraTS DEVELOPMENT set only)

Pre-declared on 2026-10-13, **before any v3 run**. This overrides the stop rule of
`IMPROVEMENT_PROTOCOL_ESRG.md` at the user's request (logged in `PROTOCOL_BRATS.md` §10).
TEST is not touched. This document is not edited in place; changes go in §8.

## 1. Principles (user decisions)
- **ESRG stays a standalone algorithm** with its own engine (passes, lazy re-check, 8-connectivity).
- **Same settings for the applied methods:** SRG with a method switched on uses ESRG's settings
  for that method (k_L, k_G, window radius, σ cap, healthy rule); SRG gets no separate tuning.
- ESRG keeps automatic seeding, the local log-domain difference measure (LLDM) and the adaptive
  stopping criterion (ASC). New pieces are kept only if they pass §5.

## 2. Changes under test (all default off; existing results unchanged)
1. **Global reference = running region mean** (`global_running`): the ASC global test
   |G − μ_R| ≤ k_G·σ uses the region mean updated per absorbed pixel instead of the mean frozen at
   the start of the pass. σ stays frozen per pass.
2. **Healthy-tissue rule** (`healthy_rule`): the healthy prototype μ_H is the median of G over the
   brain interior (background already removed by the brain mask). A pixel is *healthy-near* if
   |G − μ_H| < |G − μ_R|.
   - `veto`: healthy-near pixels are rejected.
   - `rescue` (the user's rule): a pixel that fails the local bound but passes the global bound is
     absorbed if it is not healthy-near.
   - `both`: veto and rescue.
   Prototype variants (`healthy_proto`): median (primary), histogram mode, middle class of a 3-class
   multi-Otsu.
3. **SRG fidelity fix:** SRG becomes 8-connected as in Adams & Bischof (1994), Sec. II
   (`srg_connectivity`, 4 kept for reproducing earlier results). This changes the baseline by about
   −0.003 DSC on 80 DEV slices (measured before this protocol).

## 3. Data split (fresh)
DEV (518 patients) per tumor type: files sorted, shuffled with `random.Random(2026)`, then
**D1 = first 50% (screen), D2 = next 25% (confirm), D3 = last 25% (final, looked at once)**.
Planted P1 seeds (`experiments/evaluate.py: seed_protocol`). Objective = (glioma mean DSC +
meningioma mean DSC) / 2.

## 4. Rounds (ESRG; current = v2: k 4/5, σ_cap 0.08, flood guard 0.15, r = 1)
- **Round 1** (20 configurations): `global_running` ∈ {off, on} × `healthy_rule` ∈ {off, veto,
  rescue, both} × v2 add-ons (σ cap + flood guard) ∈ {on, off} = 16; plus `healthy_proto` ∈ {mode,
  otsu3} for rule ∈ {veto, both} with running off and add-ons on = 4.
- **Round 2** (10 configurations), on the Round-1 result: (k_L, k_G) ∈ {(2, 3), (3, 4), (4, 5),
  (5, 6), (6, 8)} × add-ons ∈ {on, off}.
- **Review point:** after Round 2 the results are reported to the user before the SRG mirror arms
  (S2) and the final D3 check.

## 5. Acceptance and success (fixed now)
- Best on D1 wins (ties within 0.005: fewer changes from current, then lower k_L). **Accepted only if
  it gains ≥ 0.02 on D2 with no tumor type down by more than 0.01**; otherwise the current setting is
  kept. No second try.
- **Success:** ESRG beats plain SRG (8-connected, 3 planted seeds) by **≥ 0.02 mean DSC on D3**.
  Otherwise the effort stops and the gap is reported. Per-type results are always shown.
- Budget: the two rounds above. Any further idea needs an entry in §8 before it is run.
- Results are saved to `configs/esrg_brats_frozen_v3.json` (never overwriting v1/v2); nothing is
  frozen for TEST without the user's approval.

## 6. SRG mirrors (S2, after the review point)
SRG + LLDM, SRG + ASC, SRG + healthy rule, each with ESRG's accepted settings and SRG's own
competitors, on D1 + D2; plain SRG 4- and 8-connected as references.

## 7. Outputs
`outputs/improve_v3/`: split, per-round results, decisions, run log. Driver `experiments/improve_v3.py`.

## 8. Deviations log
- 2026-10-13 (no effect on results): the laptop slept on battery from 12:39 to 12:51 and 12:52 to
  13:48 during the reference runs. Work paused and resumed; only the `seconds` timing column of
  the runs in flight is inflated. No decision uses it.

## 9. Status at the review point (2026-10-13)
| Round | Winner on D1 | D1 objective (current 0.651) | D2 gain | Decision |
|---|---|---|---|---|
| 1 | running global mean, healthy rule off | 0.691 | +0.015 | rejected (< 0.02) |
| 2 | k_L 6.0, k_G 8.0 | 0.687 | +0.011 | rejected (< 0.02) |

ESRG therefore stays at v2. D1 observations: running global mean is the largest single effect
(+0.040); the healthy-tissue veto replaces the σ cap + flood guard as flood control (0.481 →
0.680 with them off) but adds little on top of them; rescue adds ≤ 0.004; median prototype >
mode > 3-class Otsu. Best D1 configuration 0.695 vs plain SRG 0.748 (8-connected). On D2, plain
SRG 0.711 vs ESRG v2 0.651. The SRG mirrors (§6) and the D3 check (§5) wait for the user.

## 10. Exploratory side-test (2026-10-13, user question; D1 subset of 120, nothing adopted)
SRG with LLDM and ASC switched on together, ESRG's settings, planted seeds
(`outputs/explore_combo_srg/`): mean DSC 0.732 vs plain SRG 0.736 (glioma 0.625 vs 0.652,
meningioma 0.840 vs 0.820); precision up (0.93/0.88 vs 0.83/0.82), recall down (0.55/0.86 vs
0.68/0.91); floods 0.8% vs 7.5%, under-segmentation 28% vs 14%. SRG's own tuned settings
(r = 10, k 5/7) gave 0.677. Note: "SRG + ASC alone" with ESRG's settings (0.215) is a
mis-specified arm: the σ cap 0.08 was tuned in log units and clamps σ to the noise floor in the
intensity domain. The ASC-only mirror must therefore omit the cap and flood guard (as in the earlier
tuned SRG + ASC arm), or run in the log domain.

## 11. Adoption decision (2026-10-13, by the user)
Both v3 rounds were rejected by the acceptance rule (§9). At the user's request the **running
global mean is adopted anyway**, as a deliberate, logged deviation from §5: the final ESRG is
v2 + `global_running`, healthy rule **off** (the user chose "running global mean only" after
being told that rescue was never confirmed: +0.004 on D1, inside the tie band, never run on D2).
Saved as `configs/esrg_brats_frozen_v3.json` (SRG's separately tuned settings cleared, so every
SRG mirror reads ESRG's k_L, k_G, r, cap, flood guard). The tuning is finished; D3 gets its one
look together with the dress rehearsal of the evaluation pipeline on DEV (not TEST).
Evaluation arms (`experiments/evaluate.py`): the ASC-only arms (SRG + ASC, ESRG without LLDM) omit
the σ cap, which was calibrated in log units; the flood guard (a fraction) is kept.

## 12. Correction to §10 (2026-10-13, found in the dress rehearsal)
The §10 side-test was **mislabelled**. Its arms were built from the v2 config, which still held SRG's
separately tuned settings (`srg_local_radius` = 10, `srg_k_local/k_global` = 5/7), so "SRG + LLDM
(ESRG r=1)" actually used r = 10, and "SRG + ASC / + LLDM + ASC (ESRG settings)" used k = 5/7 with
ESRG's σ cap and flood guard. The §10 numbers therefore describe those hybrid settings, not ESRG's
(r = 1, k = 4/5). The conclusion drawn there that "ESRG's settings suit the combination better than
SRG's own" is withdrawn: the comparison was cap + flood guard on (0.732) vs off (0.677).
With genuinely ESRG's settings (`configs/esrg_brats_frozen_v3.json`, SRG tuning cleared) the D3
rehearsal gives: SRG + LLDM + ASC mean DSC 0.718 vs plain SRG 0.735; SRG + LLDM alone floods
(DSC 0.208, recall 0.99, precision 0.15), so r = 1 is not safe for LLDM without ASC in SRG's engine.

