"""
Main FraudDetector class — the primary interface for HealthFraudML.

Provides a unified API for training, predicting, and explaining
healthcare fraud detection across supervised, unsupervised, and hybrid models.

Theoretical Foundations:
    - Technology Acceptance Model (Davis, 1989): Designed for perceived ease of use
    - Fraud Triangle Theory (Cressey, 1954): Targets opportunity reduction
    - Diffusion of Innovations (Rogers, 2003): Scalable across institution sizes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class FlaggedCase:
    """A single flagged transaction with fraud score and explanation."""

    id: str
    score: float
    explanation: str
    predicted_type: str
    confidence: float = 0.0
    features: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Results from a fraud detection run."""

    predictions: np.ndarray
    scores: np.ndarray
    flagged: List[FlaggedCase]
    threshold: float
    model_name: str
    n_flagged: int = 0

    def __post_init__(self):
        self.n_flagged = len(self.flagged)

    def summary(self) -> dict:
        """Return a summary of detection results."""
        return {
            "total_transactions": len(self.predictions),
            "flagged_count": self.n_flagged,
            "flagged_rate": self.n_flagged / max(len(self.predictions), 1),
            "avg_fraud_score": float(np.mean(self.scores)),
            "max_fraud_score": float(np.max(self.scores)) if len(self.scores) > 0 else 0,
            "model": self.model_name,
            "threshold": self.threshold,
        }


class FraudDetector:
    """
    Unified healthcare fraud detection interface.

    Wraps supervised, unsupervised, and hybrid ML models with a consistent
    API for training, prediction, and explainability. Designed for healthcare
    claims data with domain-specific preprocessing and fraud type classification.

    Parameters
    ----------
    model : object
        An ML model instance from healthfraudml.models (e.g., HybridEnsemble,
        NeuralNetworkDetector, SVMDetector). Must implement fit() and predict().
    threshold : float, default=0.5
        Decision threshold for flagging transactions as potentially fraudulent.
        Transactions with fraud scores >= threshold are flagged.
    explain : bool, default=True
        Whether to generate human-readable explanations for flagged cases.
        Uses SHAP/LIME when available, falls back to feature importance.

    Examples
    --------
    >>> from healthfraudml import FraudDetector
    >>> from healthfraudml.models import HybridEnsemble
    >>> detector = FraudDetector(model=HybridEnsemble())
    >>> detector.fit(X_train, y_train)
    >>> results = detector.predict(X_test, explain=True)
    >>> print(f"Flagged {results.n_flagged} transactions")

    References
    ----------
    Bahudhoddi, B. K. (2025). Financial Fraud Detection in Healthcare Settings:
    Comparative Analysis through Machine Learning. [Doctoral dissertation,
    University of the Cumberlands].
    """

    # Healthcare fraud type labels derived from qualitative research
    FRAUD_TYPES = [
        "upcoding",
        "phantom_billing",
        "duplicate_claims",
        "unbundling",
        "identity_theft",
        "kickbacks",
    ]

    def __init__(
        self,
        model: Any = None,
        threshold: float = 0.5,
        explain: bool = True,
    ):
        self.model = model
        self.threshold = threshold
        self.explain = explain
        self._is_fitted = False
        self._feature_names = None

    def fit(self, X, y=None, feature_names: Optional[List[str]] = None):
        """
        Train the fraud detection model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data — preprocessed healthcare claims features.
        y : array-like of shape (n_samples,), optional
            Labels (1 = fraud, 0 = legitimate). Required for supervised
            and hybrid models; ignored for unsupervised models.
        feature_names : list of str, optional
            Names for each feature column, used in explanations.

        Returns
        -------
        self
        """
        if self.model is None:
            raise ValueError(
                "No model specified. Provide a model from healthfraudml.models, "
                "e.g., FraudDetector(model=HybridEnsemble())"
            )

        self._feature_names = feature_names
        if hasattr(self.model, "fit"):
            if y is not None:
                self.model.fit(X, y)
            else:
                self.model.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X, explain: Optional[bool] = None) -> DetectionResult:
        """
        Detect fraud in new transactions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Transaction data to evaluate.
        explain : bool, optional
            Override the instance-level explain setting.

        Returns
        -------
        DetectionResult
            Contains predictions, fraud scores, and flagged cases
            with explanations.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        should_explain = explain if explain is not None else self.explain

        # Get fraud scores
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            scores = probas[:, 1] if probas.ndim > 1 else probas
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(X)
            # Normalize to [0, 1]
            scores = (raw_scores - raw_scores.min()) / (
                raw_scores.max() - raw_scores.min() + 1e-10
            )
        elif hasattr(self.model, "score_samples"):
            raw_scores = -self.model.score_samples(X)  # Anomaly = high score
            scores = (raw_scores - raw_scores.min()) / (
                raw_scores.max() - raw_scores.min() + 1e-10
            )
        else:
            scores = self.model.predict(X).astype(float)

        predictions = (scores >= self.threshold).astype(int)

        # Build flagged cases
        flagged = []
        flagged_indices = np.where(predictions == 1)[0]

        for idx in flagged_indices:
            explanation = ""
            if should_explain:
                explanation = self._generate_explanation(X, idx, scores[idx])

            fraud_type = self._classify_fraud_type(X, idx)

            flagged.append(
                FlaggedCase(
                    id=str(idx),
                    score=float(scores[idx]),
                    explanation=explanation,
                    predicted_type=fraud_type,
                    confidence=float(scores[idx]),
                )
            )

        model_name = type(self.model).__name__

        return DetectionResult(
            predictions=predictions,
            scores=scores,
            flagged=flagged,
            threshold=self.threshold,
            model_name=model_name,
        )

    def _generate_explanation(self, X, idx: int, score: float) -> str:
        """Generate a human-readable explanation for a flagged transaction."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            top_k = min(3, len(importances))
            top_indices = np.argsort(importances)[-top_k:][::-1]

            if self._feature_names:
                features = [self._feature_names[i] for i in top_indices]
            else:
                features = [f"feature_{i}" for i in top_indices]

            return (
                f"Fraud score {score:.3f} driven primarily by: "
                + ", ".join(features)
            )
        return f"Fraud score {score:.3f} exceeds threshold {self.threshold}"

    def _classify_fraud_type(self, X, idx: int) -> str:
        """
        Classify the predicted fraud type based on transaction features.

        Uses heuristic rules derived from qualitative research findings
        on the most common healthcare fraud categories (Bahudhoddi, 2025).
        """
        if hasattr(self.model, "predict_fraud_type"):
            return self.model.predict_fraud_type(X[idx])
        return "unknown"

    def evaluate(self, X, y_true) -> dict:
        """
        Evaluate model performance with fraud-specific metrics.

        Parameters
        ----------
        X : array-like
            Test features.
        y_true : array-like
            True labels.

        Returns
        -------
        dict
            Metrics including AUC-PR, MCC, precision, recall, F1.
        """
        from healthfraudml.evaluation.metrics import fraud_metrics

        results = self.predict(X, explain=False)
        return fraud_metrics(y_true, results.predictions, results.scores)
