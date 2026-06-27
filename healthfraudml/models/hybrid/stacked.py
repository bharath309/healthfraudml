"""
Stacked supervised-unsupervised model for healthcare fraud detection.

Carcillo et al. (2021) demonstrated that computing outlier scores at
multiple granularity levels and integrating them as features within a
supervised framework achieves significant improvements in AUC-PR.
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


class StackedDetector:
    """
    Two-stage stacked model: unsupervised anomaly features fed into
    a supervised meta-learner.

    Stage 1: Multiple unsupervised models generate anomaly scores.
    Stage 2: Supervised meta-learner combines original features with
             anomaly scores for final classification.

    Parameters
    ----------
    meta_learner : str, default="logistic"
        Meta-learner: "logistic", "rf", or "gb".
    random_state : int, default=42
    """

    def __init__(
        self,
        meta_learner: str = "logistic",
        random_state: int = 42,
    ):
        self.random_state = random_state

        # Stage 1: Unsupervised detectors
        self._isolation = IsolationForest(
            contamination=0.05, random_state=random_state, n_jobs=-1,
        )
        self._kmeans = KMeans(
            n_clusters=8, random_state=random_state, n_init=10,
        )

        # Stage 2: Meta-learner
        if meta_learner == "logistic":
            self._meta = LogisticRegression(
                class_weight="balanced", random_state=random_state,
                max_iter=1000,
            )
        elif meta_learner == "rf":
            self._meta = RandomForestClassifier(
                n_estimators=100, class_weight="balanced",
                random_state=random_state,
            )
        else:
            self._meta = GradientBoostingClassifier(
                n_estimators=100, random_state=random_state,
            )

    def fit(self, X, y):
        """Fit both stages."""
        # Stage 1
        self._isolation.fit(X)
        self._kmeans.fit(X)

        # Generate stage-1 features
        X_stacked = self._stack_features(X)

        # Stage 2
        self._meta.fit(X_stacked, y)
        return self

    def predict(self, X) -> np.ndarray:
        X_stacked = self._stack_features(X)
        return self._meta.predict(X_stacked)

    def predict_proba(self, X) -> np.ndarray:
        X_stacked = self._stack_features(X)
        if hasattr(self._meta, "predict_proba"):
            return self._meta.predict_proba(X_stacked)
        # Fallback
        preds = self._meta.predict(X_stacked)
        return np.column_stack([1 - preds, preds])

    def _stack_features(self, X) -> np.ndarray:
        """Combine original features with unsupervised anomaly scores."""
        iso_scores = self._isolation.decision_function(X)
        kmeans_labels = self._kmeans.predict(X)
        centers = self._kmeans.cluster_centers_
        kmeans_dist = np.array([
            np.linalg.norm(X[i] - centers[kmeans_labels[i]])
            for i in range(len(X))
        ])

        return np.column_stack([X, iso_scores, kmeans_dist])
