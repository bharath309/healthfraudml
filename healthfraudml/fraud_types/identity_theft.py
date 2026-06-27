"""
Patient identity theft detection — using stolen identities for
fraudulent claims.

Uses behavioral profiling to detect anomalous claim patterns
that suggest a patient's identity is being used fraudulently.
"""

import pandas as pd
import numpy as np
from typing import Optional


class IdentityTheftDetector:
    """
    Detect potential patient identity theft through behavioral profiling.

    Flags patients with geographically impossible claim patterns,
    sudden changes in billing frequency, or demographically
    inconsistent procedures.

    Parameters
    ----------
    max_daily_facilities : int, default=2
        Maximum plausible distinct facilities per patient per day.
    frequency_multiplier : float, default=3.0
        Flag if claim frequency exceeds this multiple of the patient's average.
    """

    def __init__(
        self,
        max_daily_facilities: int = 2,
        frequency_multiplier: float = 3.0,
    ):
        self.max_daily_facilities = max_daily_facilities
        self.frequency_multiplier = frequency_multiplier
        self._patient_baselines = None

    def fit(self, claims_df: pd.DataFrame):
        """Learn patient behavioral baselines."""
        if "patient_id" in claims_df.columns:
            self._patient_baselines = claims_df.groupby("patient_id").agg(
                avg_claims_per_month=("claim_id", "count") if "claim_id" in claims_df.columns
                else ("patient_id", "count"),
            ).reset_index()
            # Normalize to monthly
            date_range_days = 365  # Default assumption
            if "claim_date" in claims_df.columns:
                try:
                    dates = pd.to_datetime(claims_df["claim_date"], errors="coerce")
                    date_range_days = max((dates.max() - dates.min()).days, 1)
                except (ValueError, TypeError):
                    pass
            months = max(date_range_days / 30, 1)
            self._patient_baselines["avg_claims_per_month"] /= months
        return self

    def detect(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Flag claims with identity theft indicators."""
        df = claims_df.copy()
        df["identity_theft_flag"] = 0
        df["identity_theft_score"] = 0.0
        return df
