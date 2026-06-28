# Changelog

All notable changes to HealthFraudML will be documented in this file.

## [0.1.0] - 2026-06-27

### Added
- **Core Framework**: Healthcare claims fraud detection pipeline with modular architecture
- **BillingAuditor**: Rule-based billing audit engine covering 6 fraud patterns (upcoding, unbundling, phantom billing, duplicate claims, identity fraud, anomalous patterns)
- **LLM Integration**: Optional GPT/Claude integration for natural language fraud explanations
- **ML Comparison Suite**: Experiment script comparing 8 ML architectures:
  - Random Forest, SVM (RBF), Neural Network (MLP), Gradient Boosting, Naive Bayes, K-Means Clustering, Artificial Immune System (Negative Selection), AdaBoost-Voting Ensemble
- **Synthetic Data Generator**: Configurable healthcare claims dataset with realistic fraud distributions (50K claims, 3.2% fraud rate, 20 CPT codes, 500 providers)
- **SMOTE-ENN Resampling**: Hybrid oversampling/cleaning for severe class imbalance
- **Explainability**: SHAP-based feature importance analysis with fallback to Gini importance
- **Evaluation Metrics**: F1-score, AUC-PR, MCC (fraud-appropriate metrics, not accuracy)
- **Documentation**: Comprehensive README, contributing guidelines, and experiment instructions

### Infrastructure
- Python package structure with `setup.py` and `pip install -e .` support
- MIT License
- GitHub Issue templates for bugs and feature requests
