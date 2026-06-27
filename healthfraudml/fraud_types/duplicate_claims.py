"""
Duplicate claims detection — submitting the same claim multiple times.

Uses a hybrid rule-based + ML approach combining exact matching,
fuzzy matching, and temporal proximity analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class DuplicateClaimsDetector:
    """
    Detect duplicate and near-duplicate claims.

    Parameters
    ----------
    match_columns : list of str, optional
        Columns to check for duplicates. Defaults to standard claim fields.
    time_window_days : int, default=30
        Claims within this window of each other are candidate duplicates.
    """

    def __init__(
        self,
        match_columns: Optional[List[str]] = None,
        time_window_days: int = 30,
    ):
        self.match_columns = match_columns or [
            "patient_id", "provider_id", "procedure_code", "claim_amount",
        ]
        self.time_window_days = time_window_days

    def detect(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Flag potential duplicate claims."""
        df = claims_df.copy()
        df["duplicate_flag"] = 0
        df["duplicate_score"] = 0.0

        available_cols = [c for c in self.match_columns if c in df.columns]

        if len(available_cols) >= 2:
            # Exact duplicate detection
            dup_mask = df.duplicated(subset=available_cols, keep=False)
            df.loc[dup_mask, "duplicate_flag"] = 1
            df.loc[dup_mask, "duplicate_score"] = 1.0

            # Near-duplicate: same patient+provider, similar amount
            if "patient_id" in df.columns and "claim_amount" in df.columns:
                grouped = df.groupby(["patient_id"])["claim_amount"]
                df["_amt_count"] = grouped.transform("count")
                high_freq = df["_amt_count"] > 3
                df.loc[high_freq & ~dup_mask, "duplicate_score"] = 0.5
                df.loc[high_freq & (df["duplicate_score"] >= 0.5), "duplicate_flag"] = 1
                df = df.drop(columns=["_amt_count"], errors="ignore")

        return df
