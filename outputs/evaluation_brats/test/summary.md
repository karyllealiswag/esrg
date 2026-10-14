# BraTS evaluation results

Set: test | n = 600 patients (one slice each) | arms: 12. Calls: Holm p < .05 and |r| >= 0.1; positive median difference = enhanced better.

## E1: A_ESRG vs P_ESRG vs P_SRG

| Scope | Metric | Median A_ESRG / P_ESRG / P_SRG_solo / P_SRG | Friedman p | Kendall W | A_ESRG vs P_SRG | P_ESRG vs P_SRG | A_ESRG vs P_ESRG |
|---|---|---|---|---|---|---|---|
| glioma | DSC | 0.265 / 0.562 / 0.023 / 0.684 | < .001 | 0.53 | -0.152 (p < .001, r -0.62) worse | -0.038 (p < .001, r -0.35) worse | -0.127 (p < .001, r -0.46) worse |
| glioma | Recall | 0.172 / 0.426 / 1.000 / 0.782 | < .001 | 0.72 | -0.330 (p < .001, r -0.72) worse | -0.093 (p < .001, r -0.67) worse | -0.120 (p < .001, r -0.43) worse |
| glioma | Precision | 0.945 / 1.000 / 0.012 / 0.991 | < .001 | 0.51 | -0.014 (p < .001, r -0.43) worse | 0.000 (p < .001, r 0.37) better | -0.015 (p < .001, r -0.56) worse |
| glioma | HD95 (px) | 57.431 / 24.867 / 369.645 / 22.253 | < .001 | 0.66 | -8.504 (p < .001, r -0.40) worse | -0.966 (p .043, r -0.12) worse | -17.321 (p < .001, r -0.56) worse |
| meningioma | DSC | 0.866 / 0.876 / 0.015 / 0.931 | < .001 | 0.60 | -0.037 (p < .001, r -0.57) worse | -0.023 (p < .001, r -0.42) worse | 0.000 (p < .001, r -0.23) worse |
| meningioma | Recall | 0.840 / 0.854 / 1.000 / 0.948 | < .001 | 0.80 | -0.093 (p < .001, r -0.81) worse | -0.075 (p < .001, r -0.77) worse | 0.000 (p < .001, r -0.25) worse |
| meningioma | Precision | 0.975 / 0.984 / 0.008 / 0.955 | < .001 | 0.59 | 0.008 (p < .001, r 0.22) better | 0.015 (p < .001, r 0.54) better | 0.000 (p < .001, r -0.28) worse |
| meningioma | HD95 (px) | 9.997 / 9.074 / 375.439 / 5.996 | < .001 | 0.67 | -1.000 (p < .001, r -0.29) worse | -0.611 (p .009, r -0.15) worse | 0.000 (p < .001, r -0.22) worse |
| pooled | DSC | 0.703 / 0.788 / 0.019 / 0.852 | < .001 | 0.55 | -0.061 (p < .001, r -0.59) worse | -0.028 (p < .001, r -0.38) worse | -0.001 (p < .001, r -0.38) worse |
| pooled | Recall | 0.586 / 0.735 / 1.000 / 0.915 | < .001 | 0.76 | -0.148 (p < .001, r -0.74) worse | -0.082 (p < .001, r -0.71) worse | -0.004 (p < .001, r -0.38) worse |
| pooled | Precision | 0.965 / 0.994 / 0.010 / 0.969 | < .001 | 0.50 | 0.000 (p < .001, r -0.13) worse | 0.006 (p < .001, r 0.45) better | 0.000 (p < .001, r -0.44) worse |
| pooled | HD95 (px) | 23.345 / 16.211 / 372.216 / 13.023 | < .001 | 0.66 | -1.752 (p < .001, r -0.34) worse | -0.822 (p < .001, r -0.14) worse | -0.002 (p < .001, r -0.42) worse |

### Seed hit rate (A_ESRG)

| Scope | Hits / n | SHR | 95% CI | LB >= 70% |
|---|---|---|---|---|
| glioma | 175 / 300 | 58.3% | 52.7–63.8% | not met |
| meningioma | 255 / 300 | 85.0% | 80.5–88.6% | met |
| pooled | 430 / 600 | 71.7% | 67.9–75.1% | not met |

## E2: seeding consistency (DSC)

| Exp | Scope | Median auto / P1 / P2 / P3 | Friedman p | Kendall W | Placement SD median [IQR] | Auto not lower |
|---|---|---|---|---|---|---|
| E2a | glioma | 0.265 / 0.684 / 0.695 / 0.742 | < .001 | 0.25 | 0.030 [0.005–0.159] | no |
| E2a | meningioma | 0.866 / 0.931 / 0.923 / 0.923 | < .001 | 0.16 | 0.001 [0.000–0.005] | no |
| E2a | pooled | 0.703 / 0.852 / 0.861 / 0.863 | < .001 | 0.20 | 0.005 [0.000–0.049] | no |
| E2b | glioma | 0.370 / 0.684 / 0.695 / 0.742 | < .001 | 0.08 | 0.030 [0.005–0.159] | no |
| E2b | meningioma | 0.917 / 0.931 / 0.923 / 0.923 | < .001 | 0.02 | 0.001 [0.000–0.005] | no |
| E2b | pooled | 0.804 / 0.852 / 0.861 / 0.863 | < .001 | 0.05 | 0.005 [0.000–0.049] | no |

## E3–E6: method experiments

| Exp | What changes | Scope | Metric | Median base | Median enh | Median diff | p (Holm) | r | Call |
|---|---|---|---|---|---|---|---|---|---|
| E3 | LLDM added to SRG | glioma | Recall | 0.782 | 1.000 | 0.199 | < .001 | 0.81 | better |
| E3 | LLDM added to SRG | glioma | DSC (guard) | 0.684 | 0.133 | -0.453 | < .001 | -0.74 | worse |
| E3 | LLDM added to SRG | meningioma | Recall | 0.948 | 1.000 | 0.048 | < .001 | 0.85 | better |
| E3 | LLDM added to SRG | meningioma | DSC (guard) | 0.931 | 0.137 | -0.714 | < .001 | -0.86 | worse |
| E3 | LLDM added to SRG | pooled | Recall | 0.915 | 1.000 | 0.076 | < .001 | 0.83 | better |
| E3 | LLDM added to SRG | pooled | DSC (guard) | 0.852 | 0.134 | -0.594 | < .001 | -0.82 | worse |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | glioma | Recall | 0.406 | 0.426 | -0.001 | .665 | 0.03 | no effect |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | glioma | DSC (guard) | 0.487 | 0.562 | 0.000 | .072 | 0.12 | no effect |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | meningioma | Recall | 0.796 | 0.854 | 0.029 | < .001 | 0.47 | better |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | meningioma | DSC (guard) | 0.842 | 0.876 | 0.007 | < .001 | 0.36 | better |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | pooled | Recall | 0.609 | 0.735 | 0.014 | < .001 | 0.23 | better |
| E4 | LLDM removed from ESRG (no cap, intensity domain) | pooled | DSC (guard) | 0.673 | 0.788 | 0.004 | < .001 | 0.22 | better |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | glioma | Precision | 0.991 | 1.000 | 0.005 | < .001 | 0.62 | better |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | glioma | HD95 (px) | 22.253 | 31.012 | -2.545 | < .001 | -0.26 | worse |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | glioma | DSC (guard) | 0.684 | 0.320 | -0.140 | < .001 | -0.65 | worse |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | meningioma | Precision | 0.955 | 0.998 | 0.028 | < .001 | 0.74 | better |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | meningioma | HD95 (px) | 5.996 | 10.853 | -0.884 | .002 | -0.18 | worse |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | meningioma | DSC (guard) | 0.931 | 0.802 | -0.070 | < .001 | -0.55 | worse |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | pooled | Precision | 0.969 | 1.000 | 0.017 | < .001 | 0.68 | better |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | pooled | HD95 (px) | 13.023 | 20.607 | -1.845 | < .001 | -0.23 | worse |
| E5a | ASC replaces competitors: SRG with 3 seeds vs SRG + ASC with the tumor seed | pooled | DSC (guard) | 0.852 | 0.590 | -0.100 | < .001 | -0.61 | worse |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | glioma | Precision | 0.012 | 1.000 | 0.983 | < .001 | 0.86 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | glioma | HD95 (px) | 369.645 | 31.012 | 338.009 | < .001 | 0.86 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | glioma | DSC (guard) | 0.023 | 0.320 | 0.285 | < .001 | 0.82 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | meningioma | Precision | 0.008 | 0.998 | 0.980 | < .001 | 0.86 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | meningioma | HD95 (px) | 375.439 | 10.853 | 359.156 | < .001 | 0.86 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | meningioma | DSC (guard) | 0.015 | 0.802 | 0.788 | < .001 | 0.85 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | pooled | Precision | 0.010 | 1.000 | 0.982 | < .001 | 0.86 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | pooled | HD95 (px) | 372.216 | 20.607 | 348.859 | < .001 | 0.86 | better |
| E5b | ASC added to SRG, tumor seed only (degenerate baseline) | pooled | DSC (guard) | 0.019 | 0.590 | 0.564 | < .001 | 0.84 | better |
| E6 | ASC removed from ESRG (compete mode) | glioma | Precision | 0.073 | 1.000 | 0.872 | < .001 | 0.86 | better |
| E6 | ASC removed from ESRG (compete mode) | glioma | HD95 (px) | 185.319 | 25.049 | 154.664 | < .001 | 0.81 | better |
| E6 | ASC removed from ESRG (compete mode) | glioma | DSC (guard) | 0.136 | 0.542 | 0.325 | < .001 | 0.55 | better |
| E6 | ASC removed from ESRG (compete mode) | meningioma | Precision | 0.095 | 0.991 | 0.827 | < .001 | 0.86 | better |
| E6 | ASC removed from ESRG (compete mode) | meningioma | HD95 (px) | 159.814 | 8.563 | 138.226 | < .001 | 0.83 | better |
| E6 | ASC removed from ESRG (compete mode) | meningioma | DSC (guard) | 0.173 | 0.865 | 0.624 | < .001 | 0.81 | better |
| E6 | ASC removed from ESRG (compete mode) | pooled | Precision | 0.080 | 0.997 | 0.856 | < .001 | 0.86 | better |
| E6 | ASC removed from ESRG (compete mode) | pooled | HD95 (px) | 174.791 | 16.072 | 146.323 | < .001 | 0.82 | better |
| E6 | ASC removed from ESRG (compete mode) | pooled | DSC (guard) | 0.147 | 0.771 | 0.491 | < .001 | 0.70 | better |

## Secondary comparisons (added after the TEST run; not part of the verdicts)

| Exp | What changes | Scope | Metric | Median base | Median enh | Median diff | p (Holm) | r | Call |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Equal operator effort: one tumor click for both algorithms | glioma | DSC | 0.023 | 0.562 | 0.524 | < .001 | 0.87 | better |
| S1 | Equal operator effort: one tumor click for both algorithms | glioma | Precision | 0.012 | 1.000 | 0.977 | < .001 | 0.87 | better |
| S1 | Equal operator effort: one tumor click for both algorithms | glioma | Recall | 1.000 | 0.426 | -0.574 | < .001 | -0.86 | worse |
| S1 | Equal operator effort: one tumor click for both algorithms | meningioma | DSC | 0.015 | 0.876 | 0.857 | < .001 | 0.87 | better |
| S1 | Equal operator effort: one tumor click for both algorithms | meningioma | Precision | 0.008 | 0.984 | 0.965 | < .001 | 0.87 | better |
| S1 | Equal operator effort: one tumor click for both algorithms | meningioma | Recall | 1.000 | 0.854 | -0.146 | < .001 | -0.86 | worse |
| S1 | Equal operator effort: one tumor click for both algorithms | pooled | DSC | 0.019 | 0.788 | 0.763 | < .001 | 0.87 | better |
| S1 | Equal operator effort: one tumor click for both algorithms | pooled | Precision | 0.010 | 0.994 | 0.971 | < .001 | 0.87 | better |
| S1 | Equal operator effort: one tumor click for both algorithms | pooled | Recall | 1.000 | 0.735 | -0.265 | < .001 | -0.86 | worse |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | glioma | DSC | 0.370 | 0.265 | -0.018 | < .001 | -0.49 | worse |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | glioma | Precision | 0.881 | 0.945 | 0.000 | .435 | 0.05 | no effect |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | glioma | Recall | 0.644 | 0.172 | -0.034 | < .001 | -0.63 | worse |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | meningioma | DSC | 0.917 | 0.866 | -0.019 | < .001 | -0.49 | worse |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | meningioma | Precision | 0.940 | 0.975 | 0.007 | < .001 | 0.41 | better |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | meningioma | Recall | 0.938 | 0.840 | -0.062 | < .001 | -0.75 | worse |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | pooled | DSC | 0.804 | 0.703 | -0.019 | < .001 | -0.49 | worse |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | pooled | Precision | 0.935 | 0.965 | 0.000 | < .001 | 0.25 | better |
| S2 | Same automatic tumor seed for both algorithms (SRG with planted competing seeds) | pooled | Recall | 0.886 | 0.586 | -0.049 | < .001 | -0.69 | worse |

## Objective verdicts

| Scope | Overall | Obj 1 SHR LB>=70% | Obj 1 consistency | Obj 1 auto not lower | Obj 2 | Obj 3 |
|---|---|---|---|---|---|---|
| glioma | not met | not met | met | not met | not met | not met |
| meningioma | not met | met | met | not met | not met | not met |
| pooled | not met | not met | met | not met | not met | not met |

## Mean DSC / recall / precision by arm (pooled)

| Arm | DSC | Recall | Precision |
|---|---|---|---|
| A_ESRG | 0.542 | 0.497 | 0.696 |
| A_SRG | 0.601 | 0.669 | 0.665 |
| P_ESRG | 0.680 | 0.630 | 0.925 |
| P_ESRG_C | 0.678 | 0.622 | 0.926 |
| P_ESRG_C_noS | 0.280 | 0.978 | 0.228 |
| P_ESRG_noL | 0.616 | 0.583 | 0.903 |
| P_SRG | 0.736 | 0.802 | 0.824 |
| P_SRG_2 | 0.734 | 0.818 | 0.804 |
| P_SRG_3 | 0.744 | 0.830 | 0.804 |
| P_SRG_L | 0.241 | 0.980 | 0.187 |
| P_SRG_solo | 0.026 | 1.000 | 0.013 |
| P_SRG_solo_S | 0.529 | 0.457 | 0.966 |

## Determinism

- A_ESRG_repeat1: 600/600 masks identical
- A_SRG_repeat1: 600/600 masks identical
- A_ESRG_repeat2: 600/600 masks identical
- A_SRG_repeat2: 600/600 masks identical
