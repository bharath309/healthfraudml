# CMS–LEIE arm — summary (sensitivity_cc_2017plus_is_fraud); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0169 ± 0.0032 | 0.0089 | 0.1652 | 0.0362 | 0.0061 ± 0.0011 | 0.7872 ± 0.0217 |
| Random Forest | 5 | 0.0096 ± 0.0015 | 0.0049 | 0.2491 | 0.0317 | 0.0043 ± 0.0009 | 0.7869 ± 0.0200 |

Rank by f1 (complete models): Gradient Boosting > Random Forest

Rank by auc_pr (complete models): Gradient Boosting > Random Forest

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0527%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0046 | 0.2996 | 0.0090 | 3.442% | 3.456% | 8.67e-19 |
| 0.5 | 0.0088 | 0.1651 | 0.0168 | 0.975% | 0.984% | 1.73e-18 |
| 0.6 | 0.0121 | 0.1194 | 0.0219 | 0.516% | 0.522% | 3.47e-18 |
| 0.7 | 0.0169 | 0.0810 | 0.0279 | 0.249% | 0.253% | 3.47e-18 |
| 0.8 | 0.0242 | 0.0461 | 0.0317 | 0.098% | 0.100% | 6.94e-18 |
| 0.9 | 0.0424 | 0.0160 | 0.0232 | 0.019% | 0.020% | 6.94e-18 |
| 0.95 | 0.0569 | 0.0028 | 0.0053 | 0.002% | 0.003% | 6.94e-18 |
Grid optimum 0.79 (F1 0.0320, P 0.0235, R 0.0499, FPR 0.109%); nested 0.0298 ± 0.0054 (thresholds [np.float64(0.79), np.float64(0.79), np.float64(0.81), np.float64(0.78), np.float64(0.78)]).
