# Experiment Results

Generated: 2026-06-28 02:06

## Classification Performance (5-Fold CV)

| Model | Category | F1-Score | AUC-PR | MCC | Precision | Recall |
|-------|----------|----------|--------|-----|-----------|--------|
| Gradient Boosting | Supervised | 0.580+-0.015 | 0.631+-0.030 | 0.562+-0.016 | 0.437+-0.016 | 0.864+-0.013 |
| Neural Network (MLP) | Supervised | 0.576+-0.018 | 0.684+-0.033 | 0.555+-0.017 | 0.437+-0.023 | 0.850+-0.020 |
| Random Forest | Supervised | 0.563+-0.016 | 0.648+-0.044 | 0.547+-0.018 | 0.416+-0.015 | 0.872+-0.018 |
| AdaBoost-Voting Ensemble | Hybrid | 0.559+-0.009 | 0.587+-0.037 | 0.547+-0.010 | 0.408+-0.009 | 0.889+-0.015 |
| SVM (RBF) | Supervised | 0.534+-0.016 | 0.529+-0.036 | 0.522+-0.020 | 0.381+-0.013 | 0.891+-0.025 |
| Naive Bayes | Supervised | 0.494+-0.004 | 0.521+-0.048 | 0.478+-0.005 | 0.346+-0.007 | 0.862+-0.026 |

## Unsupervised Model Performance

**K-Means**: Silhouette=0.395, Fraud concentration=5.3%

**AIS**: Recall=0.512, Precision=0.345, FPR=0.100

## Feature Importance (SHAP TreeExplainer)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | total_reimbursement | 1.8731 |
| 2 | std_claim_duration | 0.4219 |
| 3 | mean_claim_duration | 0.3543 |
| 4 | claims_per_patient | 0.2663 |
| 5 | total_claims | 0.2530 |
| 6 | std_reimbursement | 0.2478 |
| 7 | unique_patients | 0.2352 |
| 8 | mean_deductible | 0.1520 |
| 9 | mean_chronic_conditions | 0.1349 |
| 10 | mean_age | 0.1252 |
