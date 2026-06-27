"""
HealthFraudML - Open-Source Healthcare Fraud Detection Framework.

A comprehensive, modular Python framework for detecting financial fraud
in healthcare claims using machine learning. Built from peer-reviewed
research on ML-based fraud detection across U.S. healthcare institutions.

Author: Bharath Kumar Bahudhoddi, Ph.D.
Based on doctoral research at the University of the Cumberlands.
"""

__version__ = "0.1.0"
__author__ = "Bharath Kumar Bahudhoddi"

from healthfraudml.detector import FraudDetector

__all__ = ["FraudDetector"]
