# HealthFraudML

![HealthFraudML Banner](healthfraudml_banner.jpg)

Welcome to the documentation for **HealthFraudML**, an open-source, modular Python framework designed for detecting financial fraud in U.S. healthcare claims using machine learning.

The framework is grounded in doctoral research by **Dr. Bharath Kumar Bahudhoddi** at the University of the Cumberlands, drawing on theoretical models including the Technology Acceptance Model (TAM), Fraud Triangle Theory, and Diffusion of Innovations Theory (DOI).

## Key Capabilities

*   **Healthcare-Specific Preprocessing**: Predefined pipelines built to handle ICD, CPT, and NDC billing codes, manage class imbalance, and enforce HIPAA compliance (de-identification).
*   **Modular Detector API**: A unified `FraudDetector` class for training, predicting, and evaluating machine learning models.
*   **Diverse Algorithm Implementations**:
    *   *Supervised*: Neural Networks, Support Vector Machines (SVMs), Bayesian Classifiers, Random Forests, and Gradient Boosting.
    *   *Unsupervised*: K-Means/X-Means Clustering, Distance-based Outliers, and Artificial Immune Systems (AIS).
    *   *Hybrid/Ensemble*: Adaptive Boosting, Stacked generalizers, and Majority Voting.
*   **Organizational Readiness Assessment**: Tools to survey and score (0-100) an institution's technological infrastructure and structural readiness for ML adoption.
*   **Domain-Specific Evaluation**: Benchmarking suite reporting F1-Score, Area Under Precision-Recall Curve (AUC-PR), and Matthews Correlation Coefficient (MCC).

---

## Quickstart

### Installation

Install via pip:

```bash
pip install healthfraudml
```

### Basic Detection Workflow

```python
from healthfraudml import FraudDetector
from healthfraudml.models import HybridEnsemble
from healthfraudml.preprocessing import ClaimsPreprocessor

# Load & split claims data
preprocessor = ClaimsPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.load_and_split("claims_data.csv")

# Train hybrid ensemble model
detector = FraudDetector(model=HybridEnsemble())
detector.fit(X_train, y_train)

# Run detection with explanations
results = detector.predict(X_test, explain=True)

# Print flagged cases
for case in results.flagged:
    print(f"Claim ID: {case.id} | Score: {case.score:.3f} | Reason: {case.explanation}")
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@phdthesis{bahudhoddi2025financial,
  title={Financial Fraud Detection in Healthcare Settings: Comparative Analysis through Machine Learning to Identify Problems Relating to Fraudulent Activities in Hospitals},
  author={Bahudhoddi, Bharath Kumar},
  year={2025},
  school={University of the Cumberlands}
}
```

---

## License

This project is licensed under the MIT License.
