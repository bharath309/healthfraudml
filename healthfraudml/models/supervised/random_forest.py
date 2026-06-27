"""
Random Forest for healthcare fraud detection.

Ensemble tree methods achieve 99.99% detection rates in hybrid
configurations (Sivanantham et al., 2021).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestDetector:
    """
    Random Forest fraud detector.

    Parameters
    ----------
    n_estimators : int, default=200
    max_depth : int, optional
    class_weight : str, default="balanced"
    random_state : int, default=42
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = None,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_
