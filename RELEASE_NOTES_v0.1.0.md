# HealthFraudML v0.1.0 — Initial Release

**An open-source framework for detecting healthcare billing fraud using machine learning**

## Highlights

- **Rule-based billing auditor** that catches upcoding, unbundling, phantom billing, duplicate claims, and anomalous billing patterns using CPT code analysis and provider profiling
- **8-model ML comparison suite** with a synthetic 50K-claim dataset — Random Forest, SVM, Gradient Boosting, Neural Network, Naive Bayes, K-Means, AIS, and a novel AdaBoost-Voting Ensemble
- **SMOTE-ENN resampling** for the extreme class imbalance inherent in fraud data (96.8% non-fraud)
- **SHAP explainability** so every fraud flag comes with a human-interpretable reason
- **Optional LLM integration** for natural language audit narratives via GPT or Claude

## Quick Start

```bash
pip install -e .
```

```python
from healthfraudml.auditor import BillingAuditor

auditor = BillingAuditor()
results = auditor.audit_claims(claims_df)
```

Run the ML comparison:
```bash
pip install scikit-learn imbalanced-learn shap
python experiments/run_comparison.py
```

## What's Next (v0.2.0)

- RAG-powered billing audit with retrieval from CMS/OIG guidelines
- Real-time streaming fraud detection pipeline
- Pre-trained model weights
- Interactive dashboard for audit results

## Links

- [Documentation](https://github.com/bharath309/healthfraudml)
- [Contributing Guide](CONTRIBUTING.md)
- [Issue Tracker](https://github.com/bharath309/healthfraudml/issues)

## Citation

If you use HealthFraudML in your research, please cite:

```bibtex
@software{bahudhoddi2026healthfraudml,
  author = {Bahudhoddi, Bharath Kumar},
  title = {HealthFraudML: Machine Learning Framework for Healthcare Claims Fraud Detection},
  year = {2026},
  url = {https://github.com/bharath309/healthfraudml}
}
```
