"""Tests for evaluation metrics."""

import numpy as np
from healthfraudml.evaluation.metrics import fraud_metrics


def test_fraud_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.8, 0.6, 0.2])

    metrics = fraud_metrics(y_true, y_pred, y_scores)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "mcc" in metrics
    assert "auc_pr" in metrics
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1


def test_fraud_metrics_cost():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    metrics = fraud_metrics(y_true, y_pred, cost_fp=1.0, cost_fn=10.0)
    # 1 FP * 1.0 + 1 FN * 10.0 = 11.0
    assert metrics["total_cost"] == 11.0
