"""
Gradient Boosting for healthcare fraud detection.

Gradient Boosting achieves 99.99% detection rates in hybrid fraud
detection configurations (Sivanantham et al., 2021).
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingDetector:
    """
    Gradient Boosting fraud detector.

    Parameters
    ----------
    n_estimators : int, default=200
    learning_rate : float, default=0.1
    max_depth : int, default=5
    random_state : int, default=42
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
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
