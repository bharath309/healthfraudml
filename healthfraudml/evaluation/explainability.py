"""
Explainability utilities for flagged fraud cases.

Practitioner research identified the 'black box' problem as a key
barrier to ML adoption (Bahudhoddi, 2025). This module provides
human-readable explanations using feature importance analysis.
"""

import numpy as np
from typing import List, Optional


def explain_prediction(
    model,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> dict:
    """
    Generate a human-readable explanation for a single prediction.

    Parameters
    ----------
    model : fitted model
        Must have feature_importances_ or coefs_ attribute.
    X_sample : array-like of shape (n_features,)
        Single transaction to explain.
    feature_names : list of str, optional
    top_k : int, default=5
        Number of top contributing features to report.

    Returns
    -------
    dict with 'top_features', 'explanation_text', 'feature_contributions'
    """
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    elif hasattr(model, "coefs_") and len(model.coefs_) > 0:
        importances = np.abs(model.coefs_[0]).mean(axis=1)

    if importances is None:
        return {
            "top_features": [],
            "explanation_text": "Model does not support feature importance extraction.",
            "feature_contributions": {},
        }

    n_features = len(importances)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    top_indices = np.argsort(importances)[-top_k:][::-1]

    contributions = {}
    for idx in top_indices:
        contributions[feature_names[idx]] = {
            "importance": float(importances[idx]),
            "value": float(X_sample[idx]) if idx < len(X_sample) else None,
        }

    explanation_parts = []
    for name, info in contributions.items():
        explanation_parts.append(
            f"{name} (importance: {info['importance']:.3f}, "
            f"value: {info['value']:.2f})"
        )

    return {
        "top_features": [feature_names[i] for i in top_indices],
        "explanation_text": "Key contributing features: " + "; ".join(explanation_parts),
        "feature_contributions": contributions,
    }
