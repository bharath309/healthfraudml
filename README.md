# HealthFraudML — Open-Source Healthcare Fraud Detection Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, modular Python framework for detecting financial fraud in healthcare claims using machine learning. Built from peer-reviewed research on ML-based fraud detection across U.S. healthcare institutions.

## Overview

HealthFraudML provides healthcare organizations with a ready-to-use toolkit for building, evaluating, and deploying ML-based fraud detection systems. The framework implements supervised, unsupervised, and hybrid approaches identified as most effective in the healthcare fraud detection literature.

**Key Features:**
- Pre-built pipelines for healthcare claims data preprocessing
- Implementations of 8+ ML algorithms optimized for fraud detection
- Hybrid ensemble models combining supervised and unsupervised approaches
- Organizational Readiness Assessment Tool for ML adoption planning
- Privacy-preserving data handling utilities
- Configurable alert thresholds and explainable AI outputs
- Benchmarking suite for comparing model performance

## Why This Framework?

Most healthcare fraud detection research focuses on algorithm development using benchmark datasets. HealthFraudML bridges the gap between research and practice by providing:

1. **Practitioner-informed design** — Built from qualitative research with healthcare administrators, fraud analysts, and IT professionals
2. **Organizational readiness tools** — Assessment instruments based on TAM, Fraud Triangle Theory, and Diffusion of Innovations Theory
3. **Scalable architecture** — Designed for institutions of all sizes, from community clinics to large hospital systems
4. **Explainable outputs** — Every flagged transaction includes human-readable reasoning

## Installation

```bash
pip install healthfraudml
```

Or install from source:

```bash
git clone https://github.com/bharathkb/healthfraudml.git
cd healthfraudml
pip install -e .
```

## Quick Start

```python
from healthfraudml import FraudDetector
from healthfraudml.models import HybridEnsemble
from healthfraudml.preprocessing import ClaimsPreprocessor

# Load and preprocess claims data
preprocessor = ClaimsPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.load_and_split("claims_data.csv")

# Train hybrid ensemble model
detector = FraudDetector(model=HybridEnsemble())
detector.fit(X_train, y_train)

# Detect fraud with explainable outputs
results = detector.predict(X_test, explain=True)

# View flagged transactions
for case in results.flagged:
    print(f"Claim ID: {case.id}")
    print(f"Fraud Score: {case.score:.3f}")
    print(f"Reasoning: {case.explanation}")
    print(f"Fraud Type: {case.predicted_type}")
    print("---")
```

## Project Structure

```
healthfraudml/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── healthfraudml/
│   ├── __init__.py
│   ├── detector.py              # Main FraudDetector class
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── claims.py            # Healthcare claims data preprocessing
│   │   ├── feature_engineering.py # Domain-specific feature creation
│   │   └── privacy.py           # Data anonymization utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── supervised/
│   │   │   ├── neural_network.py    # MLP for fraud detection
│   │   │   ├── svm.py               # SVM with RBF kernel
│   │   │   ├── bayesian.py          # Fuzzy Bayesian classifier
│   │   │   ├── random_forest.py     # Random Forest
│   │   │   └── gradient_boosting.py # Gradient Boosting
│   │   ├── unsupervised/
│   │   │   ├── clustering.py        # K-means / X-means anomaly detection
│   │   │   ├── outlier.py           # Distance-based outlier detection
│   │   │   └── ais.py               # Artificial Immune Systems
│   │   └── hybrid/
│   │       ├── ensemble.py          # AdaBoost + Majority Voting
│   │       └── stacked.py           # Stacked supervised-unsupervised
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Fraud-specific metrics (AUC-PR, MCC, etc.)
│   │   ├── benchmark.py         # Model comparison suite
│   │   └── explainability.py    # SHAP/LIME explanations for flagged cases
│   ├── readiness/
│   │   ├── __init__.py
│   │   ├── assessment.py        # Organizational Readiness Assessment Tool
│   │   ├── tam_survey.py        # TAM-based user acceptance survey
│   │   └── report_generator.py  # Readiness report generation
│   └── fraud_types/
│       ├── __init__.py
│       ├── upcoding.py          # Upcoding detection rules
│       ├── phantom_billing.py   # Phantom billing detection
│       ├── duplicate_claims.py  # Duplicate claims detection
│       ├── unbundling.py        # Service unbundling detection
│       └── identity_theft.py    # Patient identity theft detection
├── tests/
│   ├── test_detector.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── examples/
│   ├── basic_detection.py
│   ├── hybrid_ensemble.py
│   ├── readiness_assessment.py
│   └── benchmark_comparison.py
├── docs/
│   ├── getting_started.md
│   ├── model_guide.md
│   ├── api_reference.md
│   └── readiness_tool_guide.md
└── data/
    ├── sample_claims.csv        # Synthetic sample data
    └── fraud_patterns.json      # Known fraud pattern definitions
```

## Supported Fraud Types

Based on qualitative research with U.S. healthcare professionals, the framework targets the most common fraud categories:

| Fraud Type | Description | Detection Approach |
|-----------|-------------|-------------------|
| Upcoding | Billing for more expensive services than provided | Supervised classification |
| Phantom Billing | Billing for services never rendered | Anomaly detection |
| Duplicate Claims | Submitting the same claim multiple times | Rule-based + ML hybrid |
| Unbundling | Separately billing bundled services for higher reimbursement | Pattern analysis |
| Identity Theft | Using stolen patient identities for fraudulent claims | Behavioral profiling |
| Kickbacks | Illegal referral arrangements | Network analysis |

## Organizational Readiness Assessment

The framework includes a research-based Readiness Assessment Tool grounded in three theoretical frameworks:

```python
from healthfraudml.readiness import ReadinessAssessment

assessment = ReadinessAssessment()
report = assessment.evaluate(
    institution_size="small",      # small, medium, large
    annual_claims_volume=50000,
    existing_fraud_detection="manual",  # manual, rule_based, basic_ml
    it_staff_count=3,
    annual_fraud_budget=25000
)

print(report.readiness_score)       # 0-100
print(report.recommendations)      # Prioritized action items
print(report.implementation_roadmap)  # Phased deployment plan
```

## Benchmarking

Compare model performance across standard metrics:

```python
from healthfraudml.evaluation import Benchmark

benchmark = Benchmark(X_test, y_test)
benchmark.add_model("Neural Network", nn_model)
benchmark.add_model("Hybrid Ensemble", hybrid_model)
benchmark.add_model("Random Forest", rf_model)

results = benchmark.run()
benchmark.plot_comparison()  # Generates comparison charts
benchmark.export_report("benchmark_results.pdf")
```

## Research Foundation

This framework is built on findings from a doctoral research study examining ML adoption for healthcare fraud detection across U.S. healthcare institutions. The research methodology included:

- Semi-structured interviews with healthcare administrators, fraud analysts, and IT professionals
- Document analysis of fraud detection policies and ML implementation reports
- Theoretical grounding in TAM, Fraud Triangle Theory, and Diffusion of Innovations Theory

**Citation:**
```
Bahudhoddi, B. K. (2025). Financial Fraud Detection in Healthcare Settings:
Comparative Analysis through Machine Learning to Identify Problems Relating
to Fraudulent Activities in Hospitals. [Doctoral dissertation, University of
the Cumberlands].
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where contributions are especially valuable:
- Additional ML model implementations
- Healthcare-specific feature engineering approaches
- Privacy-preserving techniques (federated learning, differential privacy)
- Benchmarking on additional datasets
- Documentation and tutorials

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

**Bharath Kumar Bahudhoddi, Ph.D.**  
- Email: bharath.p90@gmail.com
- LinkedIn: [Your LinkedIn URL]
- Google Scholar: [Your Google Scholar URL]
