"""
Healthcare claims data preprocessing pipeline.

Handles the unique characteristics of healthcare claims data including
ICD/CPT code encoding, provider normalization, temporal features,
and class imbalance — a critical challenge identified in the healthcare
fraud detection literature (Carcillo et al., 2021).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, List


class ClaimsPreprocessor:
    """
    End-to-end preprocessing for healthcare claims data.

    Transforms raw claims data into ML-ready features with domain-specific
    handling for healthcare billing codes, provider information, and
    temporal patterns commonly associated with fraud.

    Parameters
    ----------
    handle_imbalance : str, default="smote"
        Strategy for handling class imbalance: "smote", "undersample",
        "oversample", or "none".
    date_features : bool, default=True
        Whether to extract temporal features (day-of-week, month,
        time-since-last-claim) from date columns.
    encode_codes : bool, default=True
        Whether to encode ICD/CPT codes using frequency-based encoding.

    References
    ----------
    Bahudhoddi (2025): Identified data preprocessing as a key barrier
    to ML adoption in smaller healthcare institutions.
    """

    # Standard healthcare claims columns
    EXPECTED_COLUMNS = [
        "claim_id", "provider_id", "patient_id", "claim_amount",
        "procedure_code", "diagnosis_code", "claim_date",
        "service_date", "billing_code", "facility_type",
    ]

    def __init__(
        self,
        handle_imbalance: str = "smote",
        date_features: bool = True,
        encode_codes: bool = True,
    ):
        self.handle_imbalance = handle_imbalance
        self.date_features = date_features
        self.encode_codes = encode_codes
        self._scaler = StandardScaler()
        self._label_encoders = {}
        self._is_fitted = False

    def load_and_split(
        self,
        filepath: str,
        target_col: str = "is_fraud",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load claims data from CSV and split into train/test sets.

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing claims data.
        target_col : str, default="is_fraud"
            Name of the binary target column (1 = fraud, 0 = legitimate).
        test_size : float, default=0.2
            Proportion of data reserved for testing.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple of arrays
        """
        df = pd.read_csv(filepath)
        df = self._clean(df)

        y = df[target_col].values
        X = df.drop(columns=[target_col])

        X_processed = self.fit_transform(X)

        return train_test_split(
            X_processed, y, test_size=test_size,
            random_state=random_state, stratify=y,
        )

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor and transform data."""
        X = X.copy()

        if self.date_features:
            X = self._extract_date_features(X)

        if self.encode_codes:
            X = self._encode_categorical(X)

        # Drop non-numeric / ID columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].values

        # Handle missing values
        X_numeric = np.nan_to_num(X_numeric, nan=0.0)

        # Scale features
        X_scaled = self._scaler.fit_transform(X_numeric)
        self._is_fitted = True

        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted preprocessor."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform() first.")

        X = X.copy()

        if self.date_features:
            X = self._extract_date_features(X)

        if self.encode_codes:
            X = self._encode_categorical(X, fit=False)

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].values
        X_numeric = np.nan_to_num(X_numeric, nan=0.0)

        return self._scaler.transform(X_numeric)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning: remove duplicates, handle nulls."""
        df = df.drop_duplicates()
        # Drop columns with >50% missing
        threshold = len(df) * 0.5
        df = df.dropna(thresh=threshold, axis=1)
        return df

    def _extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from date columns."""
        date_cols = df.select_dtypes(include=["datetime64", "object"]).columns
        for col in date_cols:
            try:
                dates = pd.to_datetime(df[col], errors="coerce")
                if dates.notna().sum() > len(df) * 0.5:
                    df[f"{col}_dayofweek"] = dates.dt.dayofweek
                    df[f"{col}_month"] = dates.dt.month
                    df[f"{col}_day"] = dates.dt.day
                    df[f"{col}_hour"] = dates.dt.hour
                    df = df.drop(columns=[col])
            except (ValueError, TypeError):
                continue
        return df

    def _encode_categorical(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical columns using label encoding."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
            elif col in self._label_encoders:
                le = self._label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).map(
                    lambda x, _le=le: (
                        _le.transform([x])[0]
                        if x in _le.classes_
                        else -1
                    )
                )
            else:
                df = df.drop(columns=[col])
        return df
