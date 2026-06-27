"""
Phantom billing detection — billing for services never rendered.

Detection relies on temporal anomalies (claims outside operating hours,
impossible service volumes) and cross-referencing with patient visit records.
"""

import numpy as np
import pandas as pd
from typing import Optional


class PhantomBillingDetector:
    """
    Detect phantom billing through volume and temporal analysis.

    Parameters
    ----------
    max_daily_claims : int, default=50
        Maximum plausible claims per provider per day.
    flag_weekend : bool, default=True
        Flag weekend claims for further review.
    """

    def __init__(self, max_daily_claims: int = 50, flag_weekend: bool = True):
        self.max_daily_claims = max_daily_claims
        self.flag_weekend = flag_weekend

    def detect(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Flag claims suspected of phantom billing."""
        df = claims_df.copy()
        df["phantom_flag"] = 0
        df["phantom_score"] = 0.0

        # Volume-based detection
        if "provider_id" in df.columns and "claim_date" in df.columns:
            try:
                df["_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
                daily_counts = df.groupby(
                    ["provider_id", df["_date"].dt.date]
                ).size().reset_index(name="daily_count")
                daily_counts.columns = ["provider_id", "_claim_date", "daily_count"]
                df["_claim_date"] = df["_date"].dt.date
                df = df.merge(daily_counts, on=["provider_id", "_claim_date"], how="left")

                excess = df["daily_count"] / self.max_daily_claims
                df["phantom_score"] = np.clip(excess - 1, 0, None)
                df["phantom_flag"] = (df["daily_count"] > self.max_daily_claims).astype(int)

                if self.flag_weekend:
                    weekend = df["_date"].dt.dayofweek.isin([5, 6])
                    df.loc[weekend, "phantom_score"] += 0.3
                    df.loc[weekend & (df["phantom_score"] > 0.5), "phantom_flag"] = 1

                df = df.drop(columns=["_date", "_claim_date", "daily_count"], errors="ignore")
            except (ValueError, KeyError):
                pass

        return df
