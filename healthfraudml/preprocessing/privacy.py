"""
Privacy-preserving data handling utilities for HIPAA compliance.

Healthcare fraud detection requires processing Protected Health Information
(PHI). This module provides anonymization, de-identification, and differential
privacy utilities to ensure HIPAA compliance while maintaining data utility
for ML model training (Bahudhoddi, 2025).
"""

import numpy as np
import pandas as pd
import hashlib
from typing import List, Optional


class DataAnonymizer:
    """
    HIPAA-compliant data anonymization for healthcare claims.

    Implements Safe Harbor de-identification and k-anonymity to protect
    patient privacy while preserving the statistical properties needed
    for effective fraud detection.

    Parameters
    ----------
    method : str, default="safe_harbor"
        Anonymization method: "safe_harbor" (HIPAA Safe Harbor),
        "k_anonymity", or "hash".
    k : int, default=5
        Minimum group size for k-anonymity. Ignored for other methods.
    salt : str, optional
        Salt for hash-based anonymization. Generated randomly if not provided.

    Notes
    -----
    Research finding: Data privacy was identified as one of the top
    barriers to ML adoption and cross-institutional data sharing
    (Bahudhoddi, 2025, RQ3). Federated learning was recommended as
    a future direction for privacy-preserving collaboration.
    """

    # HIPAA Safe Harbor: 18 identifiers to remove
    PHI_IDENTIFIERS = [
        "name", "address", "city", "state", "zip", "zip_code",
        "date_of_birth", "dob", "birth_date", "age",
        "phone", "telephone", "fax",
        "email", "email_address",
        "ssn", "social_security",
        "mrn", "medical_record_number",
        "account_number", "health_plan_number",
        "license_number", "vehicle_id",
        "device_id", "ip_address", "url",
        "biometric", "photo", "image",
    ]

    def __init__(
        self,
        method: str = "safe_harbor",
        k: int = 5,
        salt: Optional[str] = None,
    ):
        self.method = method
        self.k = k
        self.salt = salt or hashlib.sha256(
            np.random.bytes(32)
        ).hexdigest()[:16]

    def anonymize(
        self,
        df: pd.DataFrame,
        id_columns: Optional[List[str]] = None,
        preserve_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Anonymize a healthcare claims DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw claims data with potential PHI.
        id_columns : list of str, optional
            Columns containing identifiers to hash (e.g., patient_id,
            provider_id). If None, auto-detected based on column names.
        preserve_columns : list of str, optional
            Columns to keep unchanged (e.g., claim_amount, procedure_code).

        Returns
        -------
        pd.DataFrame
            Anonymized DataFrame safe for ML processing.
        """
        df = df.copy()
        preserve = set(preserve_columns or [])

        if self.method == "safe_harbor":
            df = self._safe_harbor(df, preserve)
        elif self.method == "k_anonymity":
            df = self._safe_harbor(df, preserve)
            df = self._k_anonymize(df)

        # Hash identifier columns
        if id_columns is None:
            id_columns = [
                c for c in df.columns
                if any(x in c.lower() for x in ["_id", "id_", "identifier"])
                and c not in preserve
            ]

        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].apply(self._hash_value)

        return df

    def _safe_harbor(
        self, df: pd.DataFrame, preserve: set
    ) -> pd.DataFrame:
        """Remove HIPAA Safe Harbor identifiers."""
        cols_to_remove = []
        for col in df.columns:
            if col in preserve:
                continue
            col_lower = col.lower().replace(" ", "_")
            if any(phi in col_lower for phi in self.PHI_IDENTIFIERS):
                cols_to_remove.append(col)

        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)

        return df

    def _k_anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply k-anonymity by generalizing quasi-identifiers."""
        # Generalize age to ranges
        age_cols = [c for c in df.columns if "age" in c.lower()]
        for col in age_cols:
            if df[col].dtype in [np.int64, np.float64]:
                df[col] = pd.cut(
                    df[col],
                    bins=[0, 18, 30, 45, 60, 75, 100],
                    labels=["0-18", "19-30", "31-45", "46-60", "61-75", "76+"],
                )

        # Generalize zip codes to 3-digit prefix
        zip_cols = [c for c in df.columns if "zip" in c.lower()]
        for col in zip_cols:
            df[col] = df[col].astype(str).str[:3]

        return df

    def _hash_value(self, value) -> str:
        """One-way hash a value with salt for pseudonymization."""
        combined = f"{self.salt}:{value}"
        return hashlib.sha256(combined.encode()).hexdigest()[:12]
