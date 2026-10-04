# CMS–LEIE arm — summary (primary_is_fraud); generated from saved fold outputs

| model | folds | F1 | precision | recall | MCC | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 5 | 0.0175 ± 0.0024 | 0.0090 | 0.2955 | 0.0476 | 0.0102 ± 0.0010 | 0.8009 ± 0.0138 |
| Random Forest | 5 | 0.0108 ± 0.0008 | 0.0055 | 0.3945 | 0.0407 | 0.0082 ± 0.0014 | 0.7991 ± 0.0162 |
| Neural Network (MLP) | 5 | 0.0030 ± 0.0001 | 0.0015 | 0.5450 | 0.0154 | 0.0019 ± 0.0001 | 0.6735 ± 0.0143 |
| Naive Bayes | 5 | 0.0025 ± 0.0012 | 0.0013 | 0.6023 | 0.0009 | 0.0013 ± 0.0001 | 0.5859 ± 0.0254 |
| SVM (RBF) | 5 | 0.0029 ± 0.0004 | 0.0015 | 0.0668 | 0.0045 | 0.0012 ± 0.0000 | 0.6161 ± 0.0101 |
| AdaBoost-Voting Ensemble | 5 | 0.0081 ± 0.0005 | 0.0041 | 0.4596 | 0.0362 | 0.0080 ± 0.0011 | 0.7948 ± 0.0152 |

Rank by f1 (complete models): Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > SVM (RBF) > Naive Bayes

Rank by auc_pr (complete models): Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > Naive Bayes > SVM (RBF)

## Gradient Boosting threshold study (pooled OOF, prevalence 0.0826%)
| threshold | precision | recall | F1 | FPR | flagged share | identity |Δ| |
|---|---|---|---|---|---|---|
| 0.3 | 0.0049 | 0.4498 | 0.0096 | 7.588% | 7.619% | 8.67e-19 |
| 0.5 | 0.0090 | 0.2956 | 0.0175 | 2.690% | 2.712% | 0.00e+00 |
| 0.6 | 0.0121 | 0.2382 | 0.0231 | 1.605% | 1.623% | 1.73e-18 |
| 0.7 | 0.0163 | 0.1809 | 0.0299 | 0.904% | 0.918% | 0.00e+00 |
| 0.8 | 0.0229 | 0.1211 | 0.0385 | 0.427% | 0.436% | 3.47e-18 |
| 0.9 | 0.0370 | 0.0545 | 0.0441 | 0.117% | 0.122% | 6.94e-18 |
| 0.95 | 0.0556 | 0.0173 | 0.0264 | 0.024% | 0.026% | 0.00e+00 |
Grid optimum 0.89 (F1 0.0443, P 0.0346, R 0.0613, FPR 0.141%); nested 0.0445 ± 0.0046 (thresholds [np.float64(0.89), np.float64(0.89), np.float64(0.89), np.float64(0.89), np.float64(0.9)]).
