"""
Model comparison benchmarking suite.

Compare multiple fraud detection models side-by-side with standardized
metrics and visualization.
"""

import numpy as np
from typing import Dict, Any
from healthfraudml.evaluation.metrics import fraud_metrics


class Benchmark:
    """
    Benchmark multiple fraud detection models on the same test set.

    Parameters
    ----------
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.

    Examples
    --------
    >>> benchmark = Benchmark(X_test, y_test)
    >>> benchmark.add_model("Random Forest", rf_model)
    >>> benchmark.add_model("Hybrid Ensemble", hybrid_model)
    >>> results = benchmark.run()
    >>> benchmark.print_comparison()
    """

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self._models: Dict[str, Any] = {}
        self._results: Dict[str, dict] = {}

    def add_model(self, name: str, model):
        """Add a trained model to the benchmark."""
        self._models[name] = model

    def run(self) -> Dict[str, dict]:
        """Run all models and compute metrics."""
        for name, model in self._models.items():
            y_pred = model.predict(self.X_test)

            y_scores = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(self.X_test)
                y_scores = probas[:, 1] if probas.ndim > 1 else probas
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(self.X_test)

            self._results[name] = fraud_metrics(
                self.y_test, y_pred, y_scores
            )

        return self._results

    def print_comparison(self):
        """Print a formatted comparison table."""
        if not self._results:
            self.run()

        key_metrics = ["precision", "recall", "f1", "mcc", "auc_pr"]
        header = f"{'Model':<25}" + "".join(
            f"{m:>12}" for m in key_metrics
        )
        print(header)
        print("-" * len(header))

        for name, metrics in self._results.items():
            row = f"{name:<25}" + "".join(
                f"{metrics.get(m, 'N/A'):>12.4f}"
                if isinstance(metrics.get(m), (int, float))
                else f"{'N/A':>12}"
                for m in key_metrics
            )
            print(row)

    def get_best_model(self, metric: str = "mcc") -> str:
        """Return the name of the best model by a given metric."""
        if not self._results:
            self.run()

        return max(
            self._results,
            key=lambda name: self._results[name].get(metric, -1),
        )
