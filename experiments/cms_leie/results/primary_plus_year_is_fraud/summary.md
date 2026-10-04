# CMS–LEIE arm — summary (primary_plus_year_is_fraud); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0245 ± 0.0031 | 0.0128 | 0.2625 | 0.0548 | 0.0126 ± 0.0017 | 0.8306 ± 0.0092 |
| Random Forest | 5 | 0.0107 ± 0.0006 | 0.0054 | 0.4223 | 0.0419 | 0.0093 ± 0.0015 | 0.8182 ± 0.0106 |

Rank by f1 (complete models): Gradient Boosting > Random Forest

Rank by auc_pr (complete models): Gradient Boosting > Random Forest

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0826%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0071 | 0.4063 | 0.0139 | 4.726% | 4.756% | 8.67e-19 |
| 0.5 | 0.0128 | 0.2625 | 0.0244 | 1.674% | 1.694% | 3.47e-18 |
| 0.6 | 0.0167 | 0.2051 | 0.0308 | 1.000% | 1.016% | 3.47e-18 |
| 0.7 | 0.0224 | 0.1543 | 0.0391 | 0.557% | 0.569% | 3.47e-18 |
| 0.8 | 0.0317 | 0.1020 | 0.0483 | 0.258% | 0.266% | 0.00e+00 |
| 0.9 | 0.0513 | 0.0437 | 0.0472 | 0.067% | 0.070% | 1.39e-17 |
| 0.95 | 0.0850 | 0.0137 | 0.0236 | 0.012% | 0.013% | 1.39e-17 |
Grid optimum 0.86 (F1 0.0510, P 0.0408, R 0.0678, FPR 0.132%); nested 0.0512 ± 0.0058 (thresholds [np.float64(0.86), np.float64(0.86), np.float64(0.86), np.float64(0.86), np.float64(0.86)]).
