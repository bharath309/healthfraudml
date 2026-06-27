"""
Fraud-specific evaluation metrics.

Standard accuracy is misleading for fraud detection due to extreme class
imbalance (~1-5% fraud rate). This module provides metrics suited to
imbalanced classification: AUC-PR, MCC, and cost-sensitive measures.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Optional


def fraud_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
) -> dict:
    """
    Compute fraud detection-specific evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True labels (1 = fraud, 0 = legitimate).
    y_pred : array-like
        Predicted labels.
    y_scores : array-like, optional
        Predicted fraud scores/probabilities for AUC metrics.
    cost_fp : float, default=1.0
        Cost of a false positive (investigating a legitimate claim).
    cost_fn : float, default=10.0
        Cost of a false negative (missing a fraudulent claim).

    Returns
    -------
    dict
        Comprehensive metrics dictionary.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "total_cost": fp * cost_fp + fn * cost_fn,
    }

    if y_scores is not None:
        try:
            metrics["auc_pr"] = average_precision_score(y_true, y_scores)
            metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics["auc_pr"] = 0.0
            metrics["auc_roc"] = 0.0

    return metrics
