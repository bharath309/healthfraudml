# CMS–LEIE arm — summary (primary_imp_is_fraud); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0172 ± 0.0026 | 0.0089 | 0.2942 | 0.0470 | 0.0101 ± 0.0019 | 0.8008 ± 0.0153 |
| Random Forest | 5 | 0.0111 ± 0.0010 | 0.0056 | 0.3844 | 0.0408 | 0.0081 ± 0.0011 | 0.7992 ± 0.0161 |

Rank by f1 (complete models): Gradient Boosting > Random Forest

Rank by auc_pr (complete models): Gradient Boosting > Random Forest

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0826%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0047 | 0.4488 | 0.0094 | 7.777% | 7.808% | 8.67e-19 |
| 0.5 | 0.0088 | 0.2942 | 0.0172 | 2.728% | 2.750% | 0.00e+00 |
| 0.6 | 0.0118 | 0.2315 | 0.0224 | 1.608% | 1.626% | 3.47e-18 |
| 0.7 | 0.0163 | 0.1773 | 0.0298 | 0.885% | 0.899% | 0.00e+00 |
| 0.8 | 0.0232 | 0.1173 | 0.0387 | 0.409% | 0.418% | 3.47e-18 |
| 0.9 | 0.0380 | 0.0505 | 0.0434 | 0.106% | 0.110% | 6.94e-18 |
| 0.95 | 0.0587 | 0.0158 | 0.0248 | 0.021% | 0.022% | 6.94e-18 |
Grid optimum 0.89 (F1 0.0442, P 0.0358, R 0.0578, FPR 0.129%); nested 0.0441 ± 0.0064 (thresholds [np.float64(0.89), np.float64(0.86), np.float64(0.89), np.float64(0.89), np.float64(0.89)]).
