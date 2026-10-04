# CMS–LEIE arm — summary (primary_is_fraud_r50); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0262 ± 0.0039 | 0.0141 | 0.1970 | 0.0498 | 0.0091 ± 0.0012 | 0.7890 ± 0.0140 |
| Random Forest | 5 | 0.0150 ± 0.0014 | 0.0077 | 0.2957 | 0.0432 | 0.0083 ± 0.0014 | 0.7933 ± 0.0153 |

Rank by f1 (complete models): Gradient Boosting > Random Forest

Rank by auc_pr (complete models): Gradient Boosting > Random Forest

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0826%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0071 | 0.3215 | 0.0139 | 3.712% | 3.736% | 0.00e+00 |
| 0.5 | 0.0140 | 0.1970 | 0.0261 | 1.150% | 1.165% | 0.00e+00 |
| 0.6 | 0.0186 | 0.1480 | 0.0330 | 0.646% | 0.658% | 3.47e-18 |
| 0.7 | 0.0242 | 0.1008 | 0.0390 | 0.336% | 0.345% | 3.47e-18 |
| 0.8 | 0.0318 | 0.0575 | 0.0409 | 0.145% | 0.149% | 0.00e+00 |
| 0.9 | 0.0488 | 0.0187 | 0.0271 | 0.030% | 0.032% | 0.00e+00 |
| 0.95 | 0.0738 | 0.0039 | 0.0075 | 0.004% | 0.004% | 1.39e-17 |
Grid optimum 0.80 (F1 0.0409, P 0.0318, R 0.0575, FPR 0.145%); nested 0.0388 ± 0.0029 (thresholds [np.float64(0.75), np.float64(0.79), np.float64(0.82), np.float64(0.75), np.float64(0.8)]).
