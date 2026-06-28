# Getting Started with HealthFraudML

This guide will walk you through the installation, data preparation, and first model evaluation using **HealthFraudML**.

## Installation

Ensure you have Python 3.8+ installed. You can install the package using:

```bash
pip install healthfraudml
```

Or install from source in editable mode:

```bash
git clone https://github.com/bharath309/healthfraudml.git
cd healthfraudml
pip install -e .
```

## Step 1: Preprocess Data

Use the `ClaimsPreprocessor` to load and prepare your clinical billing data:

```python
from healthfraudml.preprocessing import ClaimsPreprocessor

preprocessor = ClaimsPreprocessor()
# Expects a CSV containing columns: ClaimID, ProviderID, CPTCode, ClaimAmount, IsFraud
X_train, X_test, y_train, y_test = preprocessor.load_and_split("your_claims.csv")
```

## Step 2: Choose and Train a Model

```python
from healthfraudml import FraudDetector
from healthfraudml.models.supervised import RandomForestDetector

detector = FraudDetector(model=RandomForestDetector(), threshold=0.5)
detector.fit(X_train, y_train)
```

## Step 3: Run Prediction & Evaluation

```python
results = detector.predict(X_test, explain=True)
metrics = detector.evaluate(X_test, y_test)

print(f"Model MCC: {metrics['mcc']:.3f}")
```
