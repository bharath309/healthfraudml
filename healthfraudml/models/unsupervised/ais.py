"""
Artificial Immune System (AIS) for healthcare fraud detection.

Mimics the human immune system's ability to distinguish self from non-self.
Halvaiee & Akbari (2014) achieved a 25% improvement in detection speed
over conventional AI approaches using AIS.
"""

import numpy as np
from typing import Optional


class ArtificialImmuneSystem:
    """
    Negative Selection Algorithm inspired by biological immune systems.

    Learns the profile of legitimate transactions (self) and flags
    transactions that deviate significantly (non-self) as potentially
    fraudulent.

    Parameters
    ----------
    n_detectors : int, default=500
        Number of detector antibodies to generate.
    radius : float, default=0.1
        Detection radius in normalized feature space.
    contamination : float, default=0.05
        Expected proportion of fraudulent transactions.
    random_state : int, default=42
    """

    def __init__(
        self,
        n_detectors: int = 500,
        radius: float = 0.1,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_detectors = n_detectors
        self.radius = radius
        self.contamination = contamination
        self._rng = np.random.RandomState(random_state)
        self._detectors = None
        self._self_mean = None
        self._self_std = None
        self._threshold = None

    def fit(self, X, y=None):
        """Learn the self-profile from legitimate transactions."""
        # Normalize
        self._self_mean = X.mean(axis=0)
        self._self_std = X.std(axis=0) + 1e-10
        X_norm = (X - self._self_mean) / self._self_std

        # Generate detectors that don't match self
        detectors = []
        max_attempts = self.n_detectors * 20
        attempts = 0

        while len(detectors) < self.n_detectors and attempts < max_attempts:
            candidate = self._rng.uniform(-3, 3, size=X.shape[1])
            # Check if candidate is far enough from all self samples
            distances = np.linalg.norm(X_norm - candidate, axis=1)
            if distances.min() > self.radius:
                detectors.append(candidate)
            attempts += 1

        self._detectors = np.array(detectors) if detectors else X_norm[:1]

        # Set threshold from training data anomaly scores
        scores = self._anomaly_scores(X_norm)
        self._threshold = np.percentile(
            scores, 100 * (1 - self.contamination)
        )
        return self

    def predict(self, X) -> np.ndarray:
        """1 = non-self (fraud), 0 = self (legitimate)."""
        X_norm = (X - self._self_mean) / self._self_std
        scores = self._anomaly_scores(X_norm)
        return (scores > self._threshold).astype(int)

    def score_samples(self, X) -> np.ndarray:
        """Anomaly scores (higher = more anomalous)."""
        X_norm = (X - self._self_mean) / self._self_std
        return -self._anomaly_scores(X_norm)

    def _anomaly_scores(self, X_norm) -> np.ndarray:
        """Count how many detectors are activated by each sample."""
        scores = np.zeros(len(X_norm))
        for det in self._detectors:
            distances = np.linalg.norm(X_norm - det, axis=1)
            scores += (distances < self.radius).astype(float)
        return scores / max(len(self._detectors), 1)
