"""
Hybrid Ensemble model combining supervised and unsupervised fraud detection.

Randhawa et al. (2015) achieved a perfect MCC of 1.0 using AdaBoost with
majority voting. Even with 30% noise, majority voting maintained MCC of
0.942. This implementation combines multiple detection approaches for
robust healthcare fraud detection.
"""

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from typing import Optional, List


class HybridEnsemble:
    """
    Hybrid ensemble combining AdaBoost, majority voting, and anomaly
    detection for healthcare fraud.

    Implements the approach validated by Randhawa et al. (2015) and
    Sivanantham et al. (2021), combining supervised classifiers with
    unsupervised anomaly scores as additional features.

    Parameters
    ----------
    voting : str, default="soft"
        Voting strategy: "hard" (majority) or "soft" (probability averaging).
    include_anomaly_features : bool, default=True
        Whether to append unsupervised anomaly scores as extra features
        before training the voting ensemble.
    random_state : int, default=42
    """

    def __init__(
        self,
        voting: str = "soft",
        include_anomaly_features: bool = True,
        random_state: int = 42,
    ):
        self.include_anomaly_features = include_anomaly_features
        self.random_state = random_state

        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=random_state, n_jobs=-1,
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=random_state,
            )),
            ("ada", AdaBoostClassifier(
                n_estimators=100, random_state=random_state,
            )),
        ]

        self._model = VotingClassifier(
            estimators=estimators,
            voting=voting,
        )

        self._anomaly_model = None

    def fit(self, X, y):
        """Train the hybrid ensemble."""
        X_augmented = X

        if self.include_anomaly_features:
            from sklearn.ensemble import IsolationForest

            self._anomaly_model = IsolationForest(
                contamination=0.05,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self._anomaly_model.fit(X)
            anomaly_scores = self._anomaly_model.decision_function(X)
            X_augmented = np.column_stack([X, anomaly_scores])

        self._model.fit(X_augmented, y)
        return self

    def predict(self, X) -> np.ndarray:
        X_augmented = self._augment(X)
        return self._model.predict(X_augmented)

    def predict_proba(self, X) -> np.ndarray:
        X_augmented = self._augment(X)
        return self._model.predict_proba(X_augmented)

    def _augment(self, X) -> np.ndarray:
        """Add anomaly scores if configured."""
        if self.include_anomaly_features and self._anomaly_model is not None:
            anomaly_scores = self._anomaly_model.decision_function(X)
            return np.column_stack([X, anomaly_scores])
        return X

    @property
    def feature_importances_(self):
        """Average feature importances across base estimators."""
        importances = []
        for name, est in self._model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                importances.append(est.feature_importances_)
        if importances:
            # Pad to same length if anomaly feature added
            max_len = max(len(imp) for imp in importances)
            padded = [
                np.pad(imp, (0, max_len - len(imp)))
                for imp in importances
            ]
            return np.mean(padded, axis=0)
        return None
