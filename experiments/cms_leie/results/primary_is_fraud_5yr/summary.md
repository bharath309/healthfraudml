# CMS–LEIE arm — summary (primary_is_fraud_5yr); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0120 ± 0.0011 | 0.0063 | 0.1382 | 0.0278 | 0.0029 ± 0.0004 | 0.7858 ± 0.0139 |
| Random Forest | 5 | 0.0066 ± 0.0009 | 0.0033 | 0.2662 | 0.0269 | 0.0028 ± 0.0007 | 0.7946 ± 0.0164 |

Rank by f1 (complete models): Gradient Boosting > Random Forest

Rank by auc_pr (complete models): Gradient Boosting > Random Forest

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0359%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0035 | 0.2762 | 0.0070 | 2.787% | 2.796% | 0.00e+00 |
| 0.5 | 0.0062 | 0.1382 | 0.0119 | 0.792% | 0.797% | 0.00e+00 |
| 0.6 | 0.0077 | 0.0873 | 0.0142 | 0.402% | 0.405% | 8.67e-19 |
| 0.7 | 0.0084 | 0.0406 | 0.0139 | 0.172% | 0.173% | 1.73e-18 |
| 0.8 | 0.0103 | 0.0144 | 0.0120 | 0.050% | 0.050% | 1.73e-18 |
| 0.9 | 0.0075 | 0.0010 | 0.0018 | 0.005% | 0.005% | 8.67e-19 |
| 0.95 | 0.0000 | 0.0000 | 0.0000 | 0.000% | 0.000% | 0.00e+00 |
Grid optimum 0.63 (F1 0.0148, P 0.0082, R 0.0737, FPR 0.320%); nested 0.0138 ± 0.0019 (thresholds [np.float64(0.65), np.float64(0.63), np.float64(0.75), np.float64(0.75), np.float64(0.63)]).
