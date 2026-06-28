# Model Guide

**HealthFraudML** implements multiple classes of models categorized by their learning paradigms.

## Supervised Models

These models require labeled historical claims data:

*   **RandomForestDetector**: Fast, ensemble decision trees optimized for class-imbalanced datasets.
*   **GradientBoostingDetector**: Highly accurate sequential boosting classifiers.
*   **NeuralNetworkDetector**: Multilayer Perceptrons (MLP) for capturing deep non-linear patterns.
*   **SVMDetector**: Support Vector Machines (using RBF kernel).
*   **BayesianDetector**: Fuzzy Bayesian classifier for probabilistic risk scoring.

## Unsupervised Models

Ideal for detecting novel anomalies or when labeled data is missing:

*   **ClusteringDetector**: Underpinned by K-Means and X-Means to partition claims into normal vs. abnormal behavioral cohorts.
*   **OutlierDetector**: Distance-based anomaly scoring.
*   **AISDetector**: Artificial Immune System classifier inspired by biological self-tolerance learning.

## Hybrid & Ensemble Models

*   **HybridEnsemble**: Combines AdaBoost with Majority Voting to build a robust classifier that handles up to 30% label noise.
*   **StackedDetector**: Meta-learners that stack the outputs of unsupervised anomaly models as features into a supervised classifier.
