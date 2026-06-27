"""
Domain-specific feature engineering for healthcare fraud detection.

Creates features grounded in fraud patterns identified through qualitative
research with healthcare professionals (Bahudhoddi, 2025), including
provider billing behavior, claim frequency anomalies, and cross-claim
pattern indicators aligned with the Fraud Triangle Theory (Cressey, 1954).
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class FeatureEngineer:
    """
    Healthcare fraud-specific feature engineering.

    Generates derived features that capture the behavioral and statistical
    patterns most associated with healthcare fraud types: upcoding,
    phantom billing, duplicate claims, unbundling, identity theft,
    and kickback schemes.

    Parameters
    ----------
    provider_features : bool, default=True
        Compute provider-level aggregate features (billing patterns,
        peer comparisons).
    patient_features : bool, default=True
        Compute patient-level features (claim frequency, geographic spread).
    temporal_features : bool, default=True
        Compute time-based features (weekend billing, after-hours claims).
    """

    def __init__(
        self,
        provider_features: bool = True,
        patient_features: bool = True,
        temporal_features: bool = True,
    ):
        self.provider_features = provider_features
        self.patient_features = patient_features
        self.temporal_features = temporal_features
        self._provider_stats = None
        self._patient_stats = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and attach engineered features."""
        df = df.copy()

        if self.provider_features and "provider_id" in df.columns:
            df = self._add_provider_features(df, fit=True)

        if self.patient_features and "patient_id" in df.columns:
            df = self._add_patient_features(df, fit=True)

        if self.temporal_features:
            df = self._add_temporal_features(df)

        return df

    def _add_provider_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Provider-level features for detecting upcoding and phantom billing.

        Research finding: Upcoding was identified as the most common fraud
        type across institutions. Provider-level billing pattern deviations
        are a key indicator (Bahudhoddi, 2025, RQ1).
        """
        if fit and "claim_amount" in df.columns:
            self._provider_stats = df.groupby("provider_id").agg(
                provider_mean_amount=("claim_amount", "mean"),
                provider_std_amount=("claim_amount", "std"),
                provider_claim_count=("claim_amount", "count"),
                provider_total_amount=("claim_amount", "sum"),
            ).reset_index()

        if self._provider_stats is not None:
            df = df.merge(self._provider_stats, on="provider_id", how="left")

            if "claim_amount" in df.columns:
                # Z-score of claim amount relative to provider's history
                df["provider_amount_zscore"] = (
                    (df["claim_amount"] - df["provider_mean_amount"])
                    / (df["provider_std_amount"] + 1e-10)
                )

        return df

    def _add_patient_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Patient-level features for detecting identity theft and duplicate claims.

        Research finding: Identity theft and duplicate claims often show
        anomalous patterns in patient claim frequency and geographic
        distribution (Bahudhoddi, 2025, RQ1).
        """
        if fit and "claim_amount" in df.columns:
            self._patient_stats = df.groupby("patient_id").agg(
                patient_claim_count=("claim_amount", "count"),
                patient_mean_amount=("claim_amount", "mean"),
                patient_unique_providers=(
                    "provider_id", "nunique"
                ) if "provider_id" in df.columns else ("claim_amount", "count"),
            ).reset_index()

        if self._patient_stats is not None:
            df = df.merge(self._patient_stats, on="patient_id", how="left")

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based features for detecting billing anomalies.

        Research finding: Phantom billing often occurs during off-hours
        or weekends when oversight is reduced — aligning with Fraud
        Triangle Theory's 'opportunity' dimension (Bahudhoddi, 2025).
        """
        date_cols = [c for c in df.columns if "date" in c.lower()]
        for col in date_cols:
            try:
                dates = pd.to_datetime(df[col], errors="coerce")
                if dates.notna().sum() > 0:
                    df[f"{col}_is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
                    df[f"{col}_is_month_end"] = (dates.dt.day >= 28).astype(int)
                    df[f"{col}_quarter"] = dates.dt.quarter
            except (ValueError, TypeError):
                continue

        return df
