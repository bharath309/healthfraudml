# Experiment Results

Generated: 2026-06-28 02:04

## Classification Performance (5-Fold CV)

| Model | Category | F1-Score | AUC-PR | MCC | Precision | Recall |
|-------|----------|----------|--------|-----|-----------|--------|
| Gradient Boosting | Supervised | 0.672+-0.005 | 0.815+-0.011 | 0.663+-0.005 | 0.628+-0.019 | 0.724+-0.020 |
| Neural Network (MLP) | Supervised | 0.594+-0.013 | 0.805+-0.014 | 0.594+-0.011 | 0.486+-0.025 | 0.765+-0.032 |
| Random Forest | Supervised | 0.530+-0.010 | 0.812+-0.010 | 0.546+-0.009 | 0.393+-0.012 | 0.814+-0.018 |
| AdaBoost-Voting Ensemble | Hybrid | 0.497+-0.009 | 0.807+-0.014 | 0.527+-0.005 | 0.351+-0.012 | 0.855+-0.021 |
| SVM (RBF) | Supervised | 0.443+-0.008 | 0.769+-0.014 | 0.498+-0.006 | 0.290+-0.008 | 0.938+-0.017 |
| Naive Bayes | Supervised | 0.393+-0.004 | 0.683+-0.013 | 0.457+-0.003 | 0.248+-0.003 | 0.946+-0.007 |

## Unsupervised Model Performance

**K-Means**: Silhouette=0.138, Fraud concentration=62.5%

**AIS**: Recall=0.836, Precision=0.217, FPR=0.100

## Feature Importance (SHAP TreeExplainer)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | provider_claim_count | 1.7970 |
| 2 | provider_std_amount | 0.8167 |
| 3 | amount_fair_ratio | 0.5013 |
| 4 | provider_mean_amount | 0.3899 |
| 5 | specialty_deviation | 0.2486 |
| 6 | severity_amount_ratio | 0.0877 |
| 7 | amount | 0.0842 |
| 8 | patient_age | 0.0603 |
| 9 | region_enc | 0.0354 |
| 10 | num_procedures | 0.0331 |
