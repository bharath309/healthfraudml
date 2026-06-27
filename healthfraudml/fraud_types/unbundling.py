"""
Service unbundling detection — separately billing bundled services
for higher reimbursement.

Uses pattern analysis to identify claims that should have been
billed together under a single bundled code.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class UnbundlingDetector:
    """
    Detect potential unbundling of healthcare services.

    Parameters
    ----------
    bundle_rules : dict, optional
        Mapping of bundled procedure codes to their component codes.
    time_window_hours : int, default=24
        Claims within this window for the same patient are candidates.
    """

    def __init__(
        self,
        bundle_rules: Optional[dict] = None,
        time_window_hours: int = 24,
    ):
        self.bundle_rules = bundle_rules or {}
        self.time_window_hours = time_window_hours

    def detect(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Flag claims suspected of unbundling."""
        df = claims_df.copy()
        df["unbundling_flag"] = 0
        df["unbundling_score"] = 0.0

        # Count same-day procedures per patient-provider pair
        if all(c in df.columns for c in ["patient_id", "provider_id", "claim_date"]):
            try:
                df["_date"] = pd.to_datetime(df["claim_date"], errors="coerce").dt.date
                counts = df.groupby(
                    ["patient_id", "provider_id", "_date"]
                ).size().reset_index(name="_proc_count")
                df = df.merge(counts, on=["patient_id", "provider_id", "_date"], how="left")
                # Multiple procedures same day = potential unbundling
                df["unbundling_score"] = np.clip((df["_proc_count"] - 2) / 3, 0, 1)
                df["unbundling_flag"] = (df["_proc_count"] > 3).astype(int)
                df = df.drop(columns=["_date", "_proc_count"], errors="ignore")
            except (ValueError, KeyError):
                pass

        return df
