"""
Support Vector Machine model for healthcare fraud detection.

SVMs with RBF kernel consistently outperform linear and quadratic
alternatives for fraud detection, with reductions in both false positive
and false negative rates (Gold, 2014; Wolf et al., 2015).
"""

import numpy as np
from sklearn.svm import SVC
from typing import Optional


class SVMDetector:
    """
    SVM-based fraud detector with RBF kernel.

    Parameters
    ----------
    kernel : str, default="rbf"
        Kernel type. RBF recommended per literature.
    C : float, default=1.0
        Regularization parameter.
    class_weight : str or dict, default="balanced"
        Class weight handling.
    probability : bool, default=True
        Enable probability estimates.
    random_state : int, default=42
        Random seed.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        class_weight: str = "balanced",
        probability: bool = True,
        random_state: int = 42,
    ):
        self._model = SVC(
            kernel=kernel,
            C=C,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def decision_function(self, X) -> np.ndarray:
        return self._model.decision_function(X)
