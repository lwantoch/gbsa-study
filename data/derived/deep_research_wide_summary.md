# Deep-research (wide sweep) — summary

- Bootstrap B=1000, permutation B=1000, α=20.0
- GBSA-locked combo (highest mean per-target BEDROC α=20) = `igb2_di4_salt0.15_st0.0072`
- XGBoost available: False · LightGBM available: False

## P1

| model | panel BEDROC | 95% CI | perm p | n_targets |
|---|---|---|---|---|
| GBSA-locked | 0.5412 | [0.3628, 0.7171] | NA | 9 |
| SVM-RBF | 0.5121 | [0.3340, 0.6889] | 0.0769 | 9 |
| Ridge | 0.5113 | [0.3340, 0.6842] | 0.0809 | 9 |
| ElasticNet | 0.4986 | [0.3076, 0.6817] | 0.1299 | 9 |
| RandomForest | 0.4986 | [0.3018, 0.6812] | 0.1289 | 9 |
| HistGradientBoosting | 0.4540 | [0.2630, 0.6416] | 0.3656 | 9 |
| ExtraTrees | 0.4083 | [0.2096, 0.6103] | 0.7133 | 9 |

**P1 verdict:** no tested model's 95% CI lies entirely above the GBSA-locked baseline of 0.541.

## P2

| model | panel BEDROC | 95% CI | perm p | n_targets |
|---|---|---|---|---|
| GBSA-locked | 0.5412 | [0.3628, 0.7171] | NA | 9 |
| RandomForest | 0.4833 | [0.2586, 0.7092] | 0.0529 | 9 |
| HistGradientBoosting | 0.4699 | [0.2883, 0.6581] | 0.0589 | 9 |
| ExtraTrees | 0.4669 | [0.2655, 0.6680] | 0.0729 | 9 |
| SVM-RBF | 0.4667 | [0.3500, 0.5871] | 0.0629 | 9 |
| LogReg-L1 | 0.3927 | [0.2475, 0.5727] | 0.2468 | 9 |
| LogReg-L2 | 0.2972 | [0.1814, 0.4063] | 0.6883 | 9 |

**P2 verdict:** no tested model's 95% CI lies entirely above the GBSA-locked baseline of 0.541.

