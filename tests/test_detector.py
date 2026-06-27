"""Tests for the FraudDetector class."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from healthfraudml import FraudDetector
from healthfraudml.models.supervised.random_forest import RandomForestDetector
from healthfraudml.models.hybrid.ensemble import HybridEnsemble


@pytest.fixture
def fraud_dataset():
    """Generate a synthetic fraud detection dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42,
    )
    return X, y


def test_detector_with_random_forest(fraud_dataset):
    X, y = fraud_dataset
    detector = FraudDetector(model=RandomForestDetector())
    detector.fit(X, y)
    results = detector.predict(X[:100])
    assert len(results.predictions) == 100
    assert results.n_flagged >= 0
    assert 0 <= results.threshold <= 1


def test_detector_with_hybrid_ensemble(fraud_dataset):
    X, y = fraud_dataset
    detector = FraudDetector(model=HybridEnsemble())
    detector.fit(X, y)
    results = detector.predict(X[:50])
    assert len(results.predictions) == 50
    assert all(isinstance(c.explanation, str) for c in results.flagged)


def test_detector_no_model():
    detector = FraudDetector()
    with pytest.raises(ValueError, match="No model specified"):
        detector.fit(np.zeros((10, 5)), np.zeros(10))


def test_detector_not_fitted():
    detector = FraudDetector(model=RandomForestDetector())
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.predict(np.zeros((10, 5)))


def test_result_summary(fraud_dataset):
    X, y = fraud_dataset
    detector = FraudDetector(model=RandomForestDetector())
    detector.fit(X, y)
    results = detector.predict(X[:100])
    summary = results.summary()
    assert "total_transactions" in summary
    assert summary["total_transactions"] == 100
    assert "flagged_rate" in summary
