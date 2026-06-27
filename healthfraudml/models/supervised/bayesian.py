"""
Bayesian classifier for healthcare fraud detection.

Fuzzy Bayesian approaches show strong specificity (0.968) in healthcare
insurance fraud detection (Chan & Lan, 2001). Dynamic Bayesian Belief
Networks enable automated knowledge updating to adapt to shifting
fraud patterns (Ormerod et al., 2003).
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from typing import Optional


class BayesianDetector:
    """
    Gaussian Naive Bayes fraud detector with prior calibration.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance added for stability.
    prior_fraud_rate : float, optional
        Prior probability of fraud. If None, learned from data.
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        prior_fraud_rate: Optional[float] = None,
    ):
        priors = [1 - prior_fraud_rate, prior_fraud_rate] if prior_fraud_rate else None
        self._model = GaussianNB(
            var_smoothing=var_smoothing,
            priors=priors,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)
