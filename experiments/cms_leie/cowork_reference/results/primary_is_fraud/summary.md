# CMS–LEIE arm — summary (primary_is_fraud); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0177 ± 0.0026 | 0.0091 | 0.2983 | 0.0481 | 0.0101 ± 0.0016 | 0.8014 ± 0.0147 |
| Random Forest | 5 | 0.0108 ± 0.0009 | 0.0055 | 0.3939 | 0.0406 | 0.0083 ± 0.0014 | 0.7992 ± 0.0159 |
| Neural Network (MLP) | 5 | 0.0030 ± 0.0001 | 0.0015 | 0.5453 | 0.0151 | 0.0019 ± 0.0001 | 0.6694 ± 0.0112 |
| Naive Bayes | 5 | 0.0025 ± 0.0012 | 0.0013 | 0.6023 | 0.0009 | 0.0013 ± 0.0001 | 0.5859 ± 0.0254 |
| SVM (RBF) | 5 | 0.0029 ± 0.0004 | 0.0015 | 0.0667 | 0.0045 | 0.0012 ± 0.0000 | 0.6161 ± 0.0099 |
| AdaBoost-Voting Ensemble | 5 | 0.0080 ± 0.0005 | 0.0041 | 0.4575 | 0.0360 | 0.0080 ± 0.0011 | 0.7947 ± 0.0155 |

Rank by f1 (complete models): Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > SVM (RBF) > Naive Bayes
- vs paper synthetic: Kendall τ = 0.73 (p = 0.06), Spearman ρ = 0.83; paper order: Gradient Boosting > Neural Network (MLP) > Random Forest > AdaBoost-Voting Ensemble > SVM (RBF) > Naive Bayes
- vs paper kaggle_medicare: Kendall τ = 0.73 (p = 0.06), Spearman ρ = 0.83; paper order: Gradient Boosting > Neural Network (MLP) > Random Forest > AdaBoost-Voting Ensemble > SVM (RBF) > Naive Bayes

Rank by auc_pr (complete models): Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > Naive Bayes > SVM (RBF)
- vs paper synthetic: Kendall τ = 0.87 (p = 0.02), Spearman ρ = 0.94; paper order: Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > SVM (RBF) > Naive Bayes
- vs paper kaggle_medicare: Kendall τ = 0.33 (p = 0.47), Spearman ρ = 0.54; paper order: Neural Network (MLP) > Random Forest > Gradient Boosting > AdaBoost-Voting Ensemble > SVM (RBF) > Naive Bayes

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0826%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0048 | 0.4487 | 0.0096 | 7.616% | 7.646% | 0.00e+00 |
| 0.5 | 0.0091 | 0.2983 | 0.0176 | 2.694% | 2.716% | 1.73e-18 |
| 0.6 | 0.0122 | 0.2408 | 0.0232 | 1.611% | 1.630% | 0.00e+00 |
| 0.7 | 0.0163 | 0.1809 | 0.0298 | 0.905% | 0.919% | 3.47e-18 |
| 0.8 | 0.0226 | 0.1210 | 0.0381 | 0.432% | 0.441% | 3.47e-18 |
| 0.9 | 0.0360 | 0.0543 | 0.0433 | 0.120% | 0.125% | 0.00e+00 |
| 0.95 | 0.0526 | 0.0173 | 0.0260 | 0.026% | 0.027% | 6.94e-18 |
Grid optimum 0.88 (F1 0.0450, P 0.0330, R 0.0706, FPR 0.171%); nested 0.0457 ± 0.0056 (thresholds [np.float64(0.88), np.float64(0.88), np.float64(0.88), np.float64(0.88), np.float64(0.88)]).
