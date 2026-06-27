"""Tests for ML models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from healthfraudml.models.supervised.random_forest import RandomForestDetector
from healthfraudml.models.supervised.svm import SVMDetector
from healthfraudml.models.supervised.bayesian import BayesianDetector
from healthfraudml.models.unsupervised.clustering import ClusteringDetector
from healthfraudml.models.unsupervised.outlier import OutlierDetector
from healthfraudml.models.hybrid.ensemble import HybridEnsemble
from healthfraudml.models.hybrid.stacked import StackedDetector


@pytest.fixture
def dataset():
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2,
        weights=[0.9, 0.1], random_state=42,
    )
    return X, y


@pytest.mark.parametrize("ModelClass", [
    RandomForestDetector, SVMDetector, BayesianDetector,
])
def test_supervised_models(ModelClass, dataset):
    X, y = dataset
    model = ModelClass()
    model.fit(X, y)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    probas = model.predict_proba(X[:10])
    assert probas.shape == (10, 2)


@pytest.mark.parametrize("ModelClass", [
    ClusteringDetector, OutlierDetector,
])
def test_unsupervised_models(ModelClass, dataset):
    X, _ = dataset
    model = ModelClass()
    model.fit(X)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})


def test_hybrid_ensemble(dataset):
    X, y = dataset
    model = HybridEnsemble()
    model.fit(X, y)
    preds = model.predict(X[:10])
    assert len(preds) == 10


def test_stacked_detector(dataset):
    X, y = dataset
    model = StackedDetector()
    model.fit(X, y)
    preds = model.predict(X[:10])
    assert len(preds) == 10
