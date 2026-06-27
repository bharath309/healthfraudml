"""Tests for preprocessing modules."""

import numpy as np
import pandas as pd
import pytest
from healthfraudml.preprocessing.claims import ClaimsPreprocessor
from healthfraudml.preprocessing.privacy import DataAnonymizer
from healthfraudml.preprocessing.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_claims():
    np.random.seed(42)
    return pd.DataFrame({
        "claim_id": range(100),
        "provider_id": np.random.choice(["P001", "P002", "P003"], 100),
        "patient_id": np.random.choice(["PT01", "PT02", "PT03", "PT04"], 100),
        "claim_amount": np.random.lognormal(6, 1, 100),
        "procedure_code": np.random.choice(["99213", "99214", "99215"], 100),
        "claim_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "is_fraud": np.random.choice([0, 1], 100, p=[0.95, 0.05]),
    })


def test_preprocessor_fit_transform(sample_claims):
    preprocessor = ClaimsPreprocessor()
    X = sample_claims.drop(columns=["is_fraud"])
    result = preprocessor.fit_transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_claims)
    assert not np.isnan(result).any()


def test_anonymizer_safe_harbor():
    df = pd.DataFrame({
        "patient_id": ["PT001", "PT002"],
        "name": ["John Doe", "Jane Smith"],
        "email": ["john@test.com", "jane@test.com"],
        "claim_amount": [500, 1200],
        "ssn": ["123-45-6789", "987-65-4321"],
    })
    anonymizer = DataAnonymizer(method="safe_harbor")
    result = anonymizer.anonymize(df, preserve_columns=["claim_amount"])
    assert "name" not in result.columns
    assert "email" not in result.columns
    assert "ssn" not in result.columns
    assert "claim_amount" in result.columns


def test_feature_engineer(sample_claims):
    engineer = FeatureEngineer()
    result = engineer.fit_transform(sample_claims)
    assert "provider_mean_amount" in result.columns
    assert "patient_claim_count" in result.columns
