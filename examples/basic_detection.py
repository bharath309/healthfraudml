"""
Basic fraud detection example using HealthFraudML.

This example demonstrates the simplest workflow: load data,
train a model, and detect fraud with explanations.
"""

import numpy as np
from sklearn.datasets import make_classification
from healthfraudml import FraudDetector
from healthfraudml.models.supervised.random_forest import RandomForestDetector


def main():
    # Generate synthetic healthcare claims data
    # In practice, use ClaimsPreprocessor to load real data
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=12,
        n_classes=2,
        weights=[0.95, 0.05],  # 5% fraud rate
        random_state=42,
    )

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train detector
    detector = FraudDetector(
        model=RandomForestDetector(n_estimators=200),
        threshold=0.5,
    )
    detector.fit(X_train, y_train)

    # Detect fraud
    results = detector.predict(X_test, explain=True)

    # Print results
    print(f"Total transactions analyzed: {len(results.predictions)}")
    print(f"Flagged as potentially fraudulent: {results.n_flagged}")
    print(f"Flagged rate: {results.n_flagged / len(results.predictions):.1%}")
    print()

    # Show flagged cases
    for case in results.flagged[:5]:
        print(f"  Claim ID: {case.id}")
        print(f"  Fraud Score: {case.score:.3f}")
        print(f"  Explanation: {case.explanation}")
        print()

    # Evaluate
    metrics = detector.evaluate(X_test, y_test)
    print("Evaluation Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  MCC: {metrics['mcc']:.3f}")
    if "auc_pr" in metrics:
        print(f"  AUC-PR: {metrics['auc_pr']:.3f}")


if __name__ == "__main__":
    main()
