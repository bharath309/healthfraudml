"""
Clustering-based anomaly detection for healthcare fraud.

K-means and X-means clustering identify fraudulent behavioral patterns
without labeled data. Chang & Chang (2014) identified four fraud strategies
— direct attacks, luxury purchases, rapid profits, and pricing variations
— using X-means analysis.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusteringDetector:
    """
    K-means clustering for unsupervised fraud anomaly detection.

    Flags transactions whose distance to their nearest cluster centroid
    exceeds a threshold, indicating behavioral anomalies.

    Parameters
    ----------
    n_clusters : int, default=8
    contamination : float, default=0.05
        Expected proportion of fraudulent transactions.
    random_state : int, default=42
    """

    def __init__(
        self,
        n_clusters: int = 8,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self._model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self._threshold = None

    def fit(self, X, y=None):
        """Fit clusters. Labels (y) are ignored."""
        self._model.fit(X)
        distances = self._compute_distances(X)
        self._threshold = np.percentile(
            distances, 100 * (1 - self.contamination)
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Return binary predictions: 1 = anomaly/fraud, 0 = normal."""
        distances = self._compute_distances(X)
        return (distances > self._threshold).astype(int)

    def score_samples(self, X) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        return -self._compute_distances(X)

    def _compute_distances(self, X) -> np.ndarray:
        """Distance to nearest centroid."""
        centers = self._model.cluster_centers_
        labels = self._model.predict(X)
        distances = np.array([
            np.linalg.norm(X[i] - centers[labels[i]])
            for i in range(len(X))
        ])
        return distances
