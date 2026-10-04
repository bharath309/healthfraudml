# CMS–LEIE arm — summary (primary_is_fraud_mandatory); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0183 ± 0.0020 | 0.0095 | 0.3070 | 0.0516 | 0.0143 ± 0.0029 | 0.8432 ± 0.0175 |
| Random Forest | 5 | 0.0105 ± 0.0009 | 0.0053 | 0.3984 | 0.0428 | 0.0092 ± 0.0020 | 0.8466 ± 0.0166 |

Rank by f1 (complete models): Gradient Boosting > Random Forest

Rank by auc_pr (complete models): Gradient Boosting > Random Forest

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0470%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0051 | 0.4367 | 0.0101 | 4.003% | 4.021% | 0.00e+00 |
| 0.5 | 0.0094 | 0.3069 | 0.0183 | 1.515% | 1.529% | 1.73e-18 |
| 0.6 | 0.0128 | 0.2556 | 0.0244 | 0.927% | 0.939% | 1.73e-18 |
| 0.7 | 0.0180 | 0.2050 | 0.0330 | 0.527% | 0.537% | 3.47e-18 |
| 0.8 | 0.0263 | 0.1443 | 0.0445 | 0.252% | 0.258% | 3.47e-18 |
| 0.9 | 0.0437 | 0.0676 | 0.0531 | 0.070% | 0.073% | 1.39e-17 |
| 0.95 | 0.0793 | 0.0274 | 0.0407 | 0.015% | 0.016% | 1.39e-17 |
Grid optimum 0.90 (F1 0.0531, P 0.0437, R 0.0676, FPR 0.070%); nested 0.0505 ± 0.0073 (thresholds [np.float64(0.9), np.float64(0.9), np.float64(0.92), np.float64(0.9), np.float64(0.91)]).
