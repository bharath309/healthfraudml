"""Fraud type-specific detection modules."""

from healthfraudml.fraud_types.upcoding import UpcodingDetector
from healthfraudml.fraud_types.phantom_billing import PhantomBillingDetector
from healthfraudml.fraud_types.duplicate_claims import DuplicateClaimsDetector

__all__ = [
    "UpcodingDetector",
    "PhantomBillingDetector",
    "DuplicateClaimsDetector",
]
