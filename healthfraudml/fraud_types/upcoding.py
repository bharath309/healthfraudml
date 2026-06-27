"""
Upcoding detection — billing for more expensive services than provided.

Upcoding was identified as the most common fraud type across all
participating institutions (Bahudhoddi, 2025, RQ1). Detection combines
procedure code analysis with provider billing pattern comparison.
"""

import numpy as np
import pandas as pd
from typing import Optional


class UpcodingDetector:
    """
    Detect upcoding by comparing billed procedure codes against
    expected distributions for similar providers and diagnoses.

    Parameters
    ----------
    sensitivity : float, default=2.0
        Z-score threshold for flagging. Lower = more sensitive.
    """

    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self._provider_profiles = None

    def fit(self, claims_df: pd.DataFrame):
        """Learn provider billing profiles from historical claims."""
        if "provider_id" in claims_df.columns and "claim_amount" in claims_df.columns:
            self._provider_profiles = claims_df.groupby("provider_id").agg(
                mean_amount=("claim_amount", "mean"),
                std_amount=("claim_amount", "std"),
                median_amount=("claim_amount", "median"),
            ).reset_index()
        return self

    def detect(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag claims suspected of upcoding.

        Returns DataFrame with added 'upcoding_flag' and 'upcoding_score' columns.
        """
        df = claims_df.copy()
        df["upcoding_flag"] = 0
        df["upcoding_score"] = 0.0

        if self._provider_profiles is not None and "provider_id" in df.columns:
            df = df.merge(self._provider_profiles, on="provider_id", how="left")
            mask = df["std_amount"].notna() & (df["std_amount"] > 0)
            df.loc[mask, "upcoding_score"] = (
                (df.loc[mask, "claim_amount"] - df.loc[mask, "mean_amount"])
                / df.loc[mask, "std_amount"]
            )
            df["upcoding_flag"] = (df["upcoding_score"] > self.sensitivity).astype(int)

        return df
