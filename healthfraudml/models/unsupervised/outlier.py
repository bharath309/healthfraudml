"""
Distance-based outlier detection for healthcare fraud.

Implements Knox & Ng (1998) distance-based outliers and Isolation Forest
for scalable anomaly detection in large claims datasets.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    """
    Isolation Forest outlier detector for healthcare claims.

    Parameters
    ----------
    contamination : float, default=0.05
        Expected fraud rate.
    n_estimators : int, default=200
    random_state : int, default=42
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y=None):
        self._model.fit(X)
        return self

    def predict(self, X) -> np.ndarray:
        """1 = fraud/outlier, 0 = normal."""
        preds = self._model.predict(X)
        return (preds == -1).astype(int)

    def score_samples(self, X) -> np.ndarray:
        return self._model.score_samples(X)

    def decision_function(self, X) -> np.ndarray:
        return self._model.decision_function(X)
