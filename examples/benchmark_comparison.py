"""
Example: Benchmarking multiple fraud detection models.

Compares supervised, unsupervised, and hybrid models on the
same dataset using fraud-specific metrics.
"""

import numpy as np
from sklearn.datasets import make_classification
from healthfraudml.evaluation.benchmark import Benchmark
from healthfraudml.models.supervised.random_forest import RandomForestDetector
from healthfraudml.models.supervised.gradient_boosting import GradientBoostingDetector
from healthfraudml.models.supervised.svm import SVMDetector
from healthfraudml.models.hybrid.ensemble import HybridEnsemble
from healthfraudml.models.hybrid.stacked import StackedDetector


def main():
    # Generate dataset
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=12,
        n_classes=2, weights=[0.95, 0.05], random_state=42,
    )
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train models
    models = {
        "Random Forest": RandomForestDetector(),
        "Gradient Boosting": GradientBoostingDetector(),
        "SVM (RBF)": SVMDetector(),
        "Hybrid Ensemble": HybridEnsemble(),
        "Stacked Model": StackedDetector(),
    }

    benchmark = Benchmark(X_test, y_test)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        benchmark.add_model(name, model)

    # Run benchmark
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    benchmark.print_comparison()

    best = benchmark.get_best_model(metric="mcc")
    print(f"\nBest model by MCC: {best}")


if __name__ == "__main__":
    main()
