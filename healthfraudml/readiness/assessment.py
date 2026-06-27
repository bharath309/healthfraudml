"""
Organizational Readiness Assessment Tool for ML-based fraud detection.

A research-validated instrument for evaluating whether a healthcare
organization is prepared to adopt machine learning for fraud detection.
Based on doctoral research with healthcare professionals across U.S.
institutions (Bahudhoddi, 2025).

Theoretical foundations:
    - Technology Acceptance Model (Davis, 1989)
    - Fraud Triangle Theory (Cressey, 1954)
    - Diffusion of Innovations Theory (Rogers, 2003)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ReadinessReport:
    """Results from an organizational readiness assessment."""

    readiness_score: float
    readiness_level: str
    dimension_scores: Dict[str, float]
    recommendations: List[str]
    implementation_roadmap: List[dict]
    weakest_dimension: str
    strongest_dimension: str

    def summary(self) -> str:
        return (
            f"Readiness Score: {self.readiness_score:.1f}/100 "
            f"({self.readiness_level})\n"
            f"Strongest: {self.strongest_dimension}\n"
            f"Weakest: {self.weakest_dimension}\n"
            f"Top recommendation: {self.recommendations[0]}"
        )


class ReadinessAssessment:
    """
    Evaluate an organization's readiness to adopt ML for fraud detection.

    Generates a Readiness Score (0-100) across 5 weighted dimensions
    and provides prioritized recommendations.

    Parameters
    ----------
    weights : dict, optional
        Custom weights for each dimension. Defaults to research-validated
        weights: Infrastructure (25%), Data (20%), Leadership (20%),
        Staff (20%), Regulatory (15%).

    Examples
    --------
    >>> assessment = ReadinessAssessment()
    >>> report = assessment.evaluate(
    ...     institution_size="small",
    ...     annual_claims_volume=50000,
    ...     existing_fraud_detection="manual",
    ...     it_staff_count=3,
    ...     annual_fraud_budget=25000,
    ... )
    >>> print(report.readiness_score)
    >>> print(report.recommendations)
    """

    DIMENSIONS = {
        "organizational_infrastructure": {
            "weight": 0.25,
            "max_score": 30,
            "questions": 6,
        },
        "data_readiness": {
            "weight": 0.20,
            "max_score": 25,
            "questions": 5,
        },
        "leadership_culture": {
            "weight": 0.20,
            "max_score": 25,
            "questions": 5,
        },
        "staff_capability": {
            "weight": 0.20,
            "max_score": 25,
            "questions": 5,
        },
        "regulatory_environment": {
            "weight": 0.15,
            "max_score": 20,
            "questions": 4,
        },
    }

    READINESS_LEVELS = {
        (80, 101): "Ready",
        (60, 80): "Developing",
        (40, 60): "Emerging",
        (20, 40): "Early Stage",
        (0, 20): "Not Ready",
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights:
            total = sum(weights.values())
            self._weights = {k: v / total for k, v in weights.items()}
        else:
            self._weights = {
                d: info["weight"] for d, info in self.DIMENSIONS.items()
            }

    def evaluate(
        self,
        institution_size: str = "medium",
        annual_claims_volume: int = 100000,
        existing_fraud_detection: str = "manual",
        it_staff_count: int = 5,
        annual_fraud_budget: float = 50000,
        has_data_science_team: bool = False,
        data_digitized: bool = True,
        has_labeled_fraud_data: bool = False,
        leadership_support: bool = True,
        staff_tech_comfort: int = 3,
        regulatory_requirements: bool = True,
        **kwargs,
    ) -> ReadinessReport:
        """
        Evaluate readiness based on organizational parameters.

        Parameters match the 5 assessment dimensions. Scores are computed
        heuristically from the provided parameters.

        Returns
        -------
        ReadinessReport
        """
        # Score each dimension (0-100 scale per dimension)
        dim_scores = {}

        # Dimension 1: Organizational Infrastructure
        infra_score = 0
        size_map = {"large": 4, "medium": 3, "small": 2}
        infra_score += size_map.get(institution_size, 2) * 5  # 10-20
        infra_score += min(it_staff_count / 2, 5) * 4  # 0-20
        infra_score += (1 if has_data_science_team else 0) * 20
        detect_map = {"basic_ml": 20, "rule_based": 12, "manual": 5, "none": 0}
        infra_score += detect_map.get(existing_fraud_detection, 5)
        budget_score = min(annual_fraud_budget / 10000, 5) * 4
        infra_score += budget_score
        dim_scores["organizational_infrastructure"] = min(infra_score, 100)

        # Dimension 2: Data Readiness
        data_score = 0
        data_score += 25 if data_digitized else 0
        data_score += 25 if has_labeled_fraud_data else 0
        volume_score = min(annual_claims_volume / 50000, 5) * 10
        data_score += volume_score
        dim_scores["data_readiness"] = min(data_score, 100)

        # Dimension 3: Leadership & Culture
        leadership_score = 40 if leadership_support else 10
        leadership_score += staff_tech_comfort * 12
        dim_scores["leadership_culture"] = min(leadership_score, 100)

        # Dimension 4: Staff Capability
        staff_score = 0
        staff_score += min(it_staff_count * 8, 40)
        staff_score += 30 if has_data_science_team else 0
        staff_score += staff_tech_comfort * 6
        dim_scores["staff_capability"] = min(staff_score, 100)

        # Dimension 5: Regulatory Environment
        reg_score = 40 if regulatory_requirements else 15
        if institution_size == "large":
            reg_score += 30
        elif institution_size == "medium":
            reg_score += 20
        else:
            reg_score += 10
        dim_scores["regulatory_environment"] = min(reg_score, 100)

        # Weighted total
        total_score = sum(
            dim_scores[d] * self._weights[d]
            for d in dim_scores
        )

        level = "Not Ready"
        for (low, high), label in self.READINESS_LEVELS.items():
            if low <= total_score < high:
                level = label
                break

        weakest = min(dim_scores, key=dim_scores.get)
        strongest = max(dim_scores, key=dim_scores.get)

        recommendations = self._generate_recommendations(
            level, dim_scores, weakest, institution_size
        )
        roadmap = self._generate_roadmap(level, weakest)

        return ReadinessReport(
            readiness_score=total_score,
            readiness_level=level,
            dimension_scores=dim_scores,
            recommendations=recommendations,
            implementation_roadmap=roadmap,
            weakest_dimension=weakest,
            strongest_dimension=strongest,
        )

    def _generate_recommendations(
        self, level: str, scores: dict, weakest: str, size: str
    ) -> List[str]:
        """Generate prioritized recommendations."""
        recs = []

        if level in ("Not Ready", "Early Stage"):
            recs.append("Prioritize data digitization and centralization")
            recs.append("Establish basic fraud detection processes")
            recs.append("Build organizational awareness of ML potential")
        elif level == "Emerging":
            recs.append("Evaluate third-party ML fraud detection services")
            recs.append("Invest in data infrastructure improvements")
            recs.append("Identify an internal ML champion")
        elif level == "Developing":
            recs.append(f"Address {weakest.replace('_', ' ')} — your bottleneck")
            recs.append("Conduct a pilot ML project with limited scope")
            recs.append("Invest in staff training programs")
        else:
            recs.append("Evaluate in-house vs. vendor ML solutions")
            recs.append("Establish a cross-functional ML governance committee")
            recs.append("Consider contributing to industry data-sharing initiatives")

        if size == "small":
            recs.append(
                "Consider third-party ML services to bridge capability gaps"
            )

        return recs

    def _generate_roadmap(self, level: str, weakest: str) -> List[dict]:
        """Generate a phased implementation roadmap."""
        if level in ("Not Ready", "Early Stage"):
            return [
                {"phase": 1, "months": "1-6", "focus": "Foundation",
                 "actions": ["Data digitization", "Staff awareness", "Budget allocation"]},
                {"phase": 2, "months": "7-12", "focus": "Basic Detection",
                 "actions": ["Rule-based system", "Staff training", "Process documentation"]},
                {"phase": 3, "months": "13-24", "focus": "ML Pilot",
                 "actions": ["Third-party ML evaluation", "Pilot project", "ROI assessment"]},
            ]
        elif level in ("Emerging", "Developing"):
            return [
                {"phase": 1, "months": "1-3", "focus": "Preparation",
                 "actions": [f"Strengthen {weakest.replace('_', ' ')}", "Vendor evaluation", "Data audit"]},
                {"phase": 2, "months": "4-6", "focus": "Pilot",
                 "actions": ["Limited-scope ML deployment", "Staff training", "Baseline metrics"]},
                {"phase": 3, "months": "7-12", "focus": "Scale",
                 "actions": ["Full deployment", "Continuous monitoring", "ROI reporting"]},
            ]
        else:
            return [
                {"phase": 1, "months": "1-2", "focus": "Selection",
                 "actions": ["Solution evaluation", "Governance setup", "Integration planning"]},
                {"phase": 2, "months": "3-4", "focus": "Deployment",
                 "actions": ["Production deployment", "Staff onboarding", "Alert calibration"]},
                {"phase": 3, "months": "5+", "focus": "Optimization",
                 "actions": ["Model retraining", "Cross-institutional collaboration", "Advanced analytics"]},
            ]
