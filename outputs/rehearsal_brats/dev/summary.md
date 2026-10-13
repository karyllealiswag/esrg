# BraTS evaluation results

Set: dev | n = 129 patients (one slice each) | arms: 14. Calls: Holm p < .05 and |r| >= 0.1; positive median difference = enhanced better.

## E1: A_ESRG vs P_ESRG vs P_SRG

| Scope | Metric | Median A_ESRG / P_ESRG / P_SRG | Friedman p | Kendall W | A_ESRG vs P_SRG | P_ESRG vs P_SRG | A_ESRG vs P_ESRG |
|---|---|---|---|---|---|---|---|
| glioma | DSC | 0.227 / 0.648 / 0.706 | < .001 | 0.23 | -0.131 (p < .001, r -0.65) worse | -0.017 (p .259, r -0.13) no effect | -0.211 (p < .001, r -0.45) worse |
| glioma | Recall | 0.201 / 0.522 / 0.803 | < .001 | 0.39 | -0.313 (p < .001, r -0.78) worse | -0.091 (p < .001, r -0.64) worse | -0.149 (p < .001, r -0.42) worse |
| glioma | Precision | 0.858 / 0.992 / 0.949 | < .001 | 0.23 | -0.010 (p .004, r -0.34) worse | 0.005 (p < .001, r 0.47) better | -0.043 (p < .001, r -0.61) worse |
| glioma | HD95 (px) | 81.123 / 25.864 / 23.755 | < .001 | 0.12 | -3.220 (p .017, r -0.31) worse | -0.893 (p .895, r 0.02) no effect | -23.463 (p < .001, r -0.56) worse |
| meningioma | DSC | 0.887 / 0.915 / 0.940 | < .001 | 0.21 | -0.042 (p < .001, r -0.66) worse | -0.013 (p .003, r -0.40) worse | 0.000 (p < .001, r -0.47) worse |
| meningioma | Recall | 0.827 / 0.882 / 0.956 | < .001 | 0.71 | -0.101 (p < .001, r -0.86) worse | -0.064 (p < .001, r -0.82) worse | 0.000 (p < .001, r -0.46) worse |
| meningioma | Precision | 0.983 / 0.987 / 0.938 | < .001 | 0.38 | 0.011 (p .002, r 0.44) better | 0.025 (p < .001, r 0.78) better | 0.000 (p .229, r -0.16) no effect |
| meningioma | HD95 (px) | 7.752 / 5.770 / 5.000 | .074 | 0.05 | -1.000 (p .035, r -0.31) worse | -0.123 (p .954, r -0.01) no effect | 0.000 (p .008, r -0.40) worse |
| pooled | DSC | 0.703 / 0.805 / 0.830 | < .001 | 0.20 | -0.075 (p < .001, r -0.65) worse | -0.014 (p .016, r -0.21) worse | -0.025 (p < .001, r -0.45) worse |
| pooled | Recall | 0.649 / 0.776 / 0.934 | < .001 | 0.51 | -0.145 (p < .001, r -0.80) worse | -0.071 (p < .001, r -0.70) worse | -0.014 (p < .001, r -0.42) worse |
| pooled | Precision | 0.962 / 0.987 / 0.940 | < .001 | 0.17 | 0.002 (p .525, r -0.06) no effect | 0.014 (p < .001, r 0.57) better | 0.000 (p < .001, r -0.48) worse |
| pooled | HD95 (px) | 25.671 / 16.279 / 16.125 | < .001 | 0.08 | -2.006 (p .001, r -0.30) worse | -0.258 (p .835, r -0.02) no effect | -0.617 (p < .001, r -0.49) worse |

### Seed hit rate (A_ESRG)

| Scope | Hits / n | SHR | 95% CI | LB >= 70% |
|---|---|---|---|---|
| glioma | 41 / 72 | 56.9% | 45.4–67.7% | not met |
| meningioma | 51 / 57 | 89.5% | 78.9–95.1% | met |
| pooled | 92 / 129 | 71.3% | 63.0–78.4% | not met |

## E2: seeding consistency (DSC)

| Exp | Scope | Median auto / P1 / P2 / P3 | Friedman p | Kendall W | Placement SD median [IQR] | Auto not lower |
|---|---|---|---|---|---|---|
| E2a | glioma | 0.227 / 0.706 / 0.737 / 0.723 | < .001 | 0.32 | 0.020 [0.001–0.248] | no |
| E2a | meningioma | 0.887 / 0.940 / 0.942 / 0.941 | < .001 | 0.16 | 0.001 [0.000–0.003] | no |
| E2a | pooled | 0.703 / 0.830 / 0.869 / 0.854 | < .001 | 0.23 | 0.004 [0.001–0.068] | no |
| E2b | glioma | 0.284 / 0.706 / 0.737 / 0.723 | < .001 | 0.13 | 0.020 [0.001–0.248] | no |
| E2b | meningioma | 0.928 / 0.940 / 0.942 / 0.941 | .361 | 0.02 | 0.001 [0.000–0.003] | yes |
| E2b | pooled | 0.845 / 0.830 / 0.869 / 0.854 | .002 | 0.04 | 0.004 [0.001–0.068] | no |

## E3–E7: method experiments

| Exp | What changes | Scope | Metric | Median base | Median enh | Median diff | p (Holm) | r | Call |
|---|---|---|---|---|---|---|---|---|---|
| E3 | LLDM added to SRG | glioma | Recall | 0.803 | 1.000 | 0.180 | < .001 | 0.80 | better |
| E3 | LLDM added to SRG | glioma | DSC (guard) | 0.706 | 0.099 | -0.523 | < .001 | -0.81 | worse |
| E3 | LLDM added to SRG | meningioma | Recall | 0.956 | 1.000 | 0.043 | < .001 | 0.85 | better |
| E3 | LLDM added to SRG | meningioma | DSC (guard) | 0.940 | 0.129 | -0.722 | < .001 | -0.87 | worse |
| E3 | LLDM added to SRG | pooled | Recall | 0.934 | 1.000 | 0.064 | < .001 | 0.82 | better |
| E3 | LLDM added to SRG | pooled | DSC (guard) | 0.830 | 0.113 | -0.621 | < .001 | -0.84 | worse |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | glioma | Recall | 0.524 | 0.522 | 0.004 | .794 | 0.04 | no effect |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | glioma | DSC (guard) | 0.571 | 0.648 | 0.008 | .794 | 0.10 | no effect |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | meningioma | Recall | 0.745 | 0.882 | 0.064 | < .001 | 0.60 | better |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | meningioma | DSC (guard) | 0.822 | 0.915 | 0.051 | < .001 | 0.60 | better |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | pooled | Recall | 0.640 | 0.776 | 0.025 | .001 | 0.29 | better |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | pooled | DSC (guard) | 0.719 | 0.805 | 0.020 | .001 | 0.31 | better |
| E5a | ASC added to SRG, with competitors | glioma | Precision | 0.949 | 0.997 | 0.000 | < .001 | 0.53 | better |
| E5a | ASC added to SRG, with competitors | glioma | HD95 (px) | 23.755 | 21.432 | 0.000 | .051 | 0.26 | no effect |
| E5a | ASC added to SRG, with competitors | glioma | DSC (guard) | 0.706 | 0.699 | 0.000 | .793 | -0.03 | no effect |
| E5a | ASC added to SRG, with competitors | meningioma | Precision | 0.938 | 0.995 | 0.015 | < .001 | 0.75 | better |
| E5a | ASC added to SRG, with competitors | meningioma | HD95 (px) | 5.000 | 10.015 | 0.000 | .108 | -0.21 | no effect |
| E5a | ASC added to SRG, with competitors | meningioma | DSC (guard) | 0.940 | 0.848 | -0.036 | < .001 | -0.59 | worse |
| E5a | ASC added to SRG, with competitors | pooled | Precision | 0.940 | 0.995 | 0.002 | < .001 | 0.62 | better |
| E5a | ASC added to SRG, with competitors | pooled | HD95 (px) | 16.125 | 16.125 | 0.000 | .575 | 0.05 | no effect |
| E5a | ASC added to SRG, with competitors | pooled | DSC (guard) | 0.830 | 0.763 | 0.000 | .001 | -0.30 | worse |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | glioma | Precision | 0.012 | 1.000 | 0.984 | < .001 | 0.87 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | glioma | HD95 (px) | 370.322 | 29.308 | 335.585 | < .001 | 0.87 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | meningioma | Precision | 0.009 | 0.998 | 0.981 | < .001 | 0.87 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | meningioma | HD95 (px) | 366.455 | 13.483 | 356.036 | < .001 | 0.87 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | pooled | Precision | 0.009 | 1.000 | 0.983 | < .001 | 0.87 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | pooled | HD95 (px) | 369.960 | 19.983 | 345.419 | < .001 | 0.87 | better |
| E6 | ASC removed from ESRG (compete mode) | glioma | Precision | 0.063 | 0.992 | 0.876 | < .001 | 0.87 | better |
| E6 | ASC removed from ESRG (compete mode) | glioma | HD95 (px) | 186.924 | 24.625 | 157.126 | < .001 | 0.85 | better |
| E6 | ASC removed from ESRG (compete mode) | glioma | DSC (guard) | 0.118 | 0.633 | 0.503 | < .001 | 0.71 | better |
| E6 | ASC removed from ESRG (compete mode) | meningioma | Precision | 0.107 | 0.995 | 0.830 | < .001 | 0.87 | better |
| E6 | ASC removed from ESRG (compete mode) | meningioma | HD95 (px) | 155.536 | 7.071 | 137.244 | < .001 | 0.87 | better |
| E6 | ASC removed from ESRG (compete mode) | meningioma | DSC (guard) | 0.194 | 0.868 | 0.622 | < .001 | 0.84 | better |
| E6 | ASC removed from ESRG (compete mode) | pooled | Precision | 0.066 | 0.994 | 0.864 | < .001 | 0.87 | better |
| E6 | ASC removed from ESRG (compete mode) | pooled | HD95 (px) | 176.173 | 17.069 | 150.413 | < .001 | 0.85 | better |
| E6 | ASC removed from ESRG (compete mode) | pooled | DSC (guard) | 0.124 | 0.771 | 0.558 | < .001 | 0.78 | better |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | glioma | DSC | 0.706 | 0.662 | -0.001 | 1.000 | -0.06 | no effect |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | glioma | Recall | 0.803 | 0.580 | -0.075 | < .001 | -0.50 | worse |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | glioma | Precision | 0.949 | 0.978 | 0.004 | < .001 | 0.42 | better |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | glioma | HD95 (px) | 23.755 | 24.484 | -0.497 | 1.000 | 0.07 | no effect |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | meningioma | DSC | 0.940 | 0.928 | -0.004 | .212 | -0.21 | no effect |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | meningioma | Recall | 0.956 | 0.923 | -0.031 | < .001 | -0.58 | worse |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | meningioma | Precision | 0.938 | 0.975 | 0.010 | < .001 | 0.48 | better |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | meningioma | HD95 (px) | 5.000 | 4.207 | 0.035 | .519 | 0.09 | no effect |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | pooled | DSC | 0.830 | 0.821 | -0.004 | .566 | -0.09 | no effect |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | pooled | Recall | 0.934 | 0.832 | -0.045 | < .001 | -0.51 | worse |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | pooled | Precision | 0.940 | 0.978 | 0.008 | < .001 | 0.43 | better |
| E7 | LLDM + ASC added to SRG together (mirror of ESRG's methods) | pooled | HD95 (px) | 16.125 | 16.031 | 0.000 | .599 | 0.05 | no effect |

## Objective verdicts

| Scope | Overall | Obj 1 SHR LB>=70% | Obj 1 consistency | Obj 1 auto not lower | Obj 2 | Obj 3 |
|---|---|---|---|---|---|---|
| glioma | not met | not met | met | not met | not met | not met |
| meningioma | not met | met | met | not met | not met | not met |
| pooled | not met | not met | met | not met | not met | not met |

## Mean DSC / recall / precision by arm (pooled)

| Arm | DSC | Recall | Precision |
|---|---|---|---|
| A_ESRG | 0.538 | 0.512 | 0.677 |
| A_SRG | 0.590 | 0.655 | 0.628 |
| P_ESRG | 0.707 | 0.669 | 0.907 |
| P_ESRG_C | 0.692 | 0.638 | 0.914 |
| P_ESRG_C_noS | 0.231 | 0.988 | 0.168 |
| P_ESRG_noL | 0.647 | 0.615 | 0.892 |
| P_SRG | 0.735 | 0.837 | 0.784 |
| P_SRG_2 | 0.762 | 0.849 | 0.798 |
| P_SRG_3 | 0.749 | 0.840 | 0.797 |
| P_SRG_L | 0.208 | 0.993 | 0.151 |
| P_SRG_LS | 0.718 | 0.697 | 0.902 |
| P_SRG_S | 0.681 | 0.618 | 0.924 |
| P_SRG_solo | 0.026 | 1.000 | 0.013 |
| P_SRG_solo_S | 0.526 | 0.441 | 0.969 |

## Determinism

- A_ESRG_repeat1: 129/129 masks identical
- A_SRG_repeat1: 129/129 masks identical
- A_ESRG_repeat2: 129/129 masks identical
- A_SRG_repeat2: 129/129 masks identical
