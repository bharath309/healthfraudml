"""
TAM-based user acceptance survey for ML fraud detection systems.

Measures perceived usefulness and perceived ease of use (Davis, 1989)
to predict staff adoption of ML-based fraud detection tools.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TAMResult:
    """Results from a TAM survey."""
    perceived_usefulness: float
    perceived_ease_of_use: float
    behavioral_intention: float
    adoption_risk: str  # "low", "medium", "high"
    recommendations: List[str]


class TAMSurvey:
    """
    Technology Acceptance Model survey instrument.

    Assesses staff readiness to adopt ML fraud detection tools
    based on perceived usefulness and perceived ease of use.
    """

    PU_QUESTIONS = [
        "ML fraud detection would improve my job performance",
        "ML fraud detection would increase my productivity",
        "ML fraud detection would enhance my effectiveness",
        "ML fraud detection would be useful in my job",
    ]

    PEOU_QUESTIONS = [
        "Learning to use ML fraud detection tools would be easy",
        "I would find ML fraud detection tools easy to use",
        "Interacting with ML tools would be clear and understandable",
        "It would be easy to become skillful with ML tools",
    ]

    def evaluate(self, pu_scores: List[int], peou_scores: List[int]) -> TAMResult:
        """
        Evaluate TAM scores (each item scored 1-7 Likert scale).

        Parameters
        ----------
        pu_scores : list of int
            Perceived Usefulness scores (1-7 each).
        peou_scores : list of int
            Perceived Ease of Use scores (1-7 each).
        """
        pu_avg = sum(pu_scores) / max(len(pu_scores), 1)
        peou_avg = sum(peou_scores) / max(len(peou_scores), 1)
        bi = 0.6 * pu_avg + 0.4 * peou_avg  # Weighted behavioral intention

        if bi >= 5.0:
            risk = "low"
            recs = ["Staff are receptive — proceed with implementation"]
        elif bi >= 3.5:
            risk = "medium"
            recs = [
                "Invest in training to improve perceived ease of use",
                "Demonstrate ROI to increase perceived usefulness",
            ]
        else:
            risk = "high"
            recs = [
                "Address resistance through change management",
                "Start with a small pilot group of willing early adopters",
                "Provide hands-on training before full rollout",
            ]

        return TAMResult(
            perceived_usefulness=pu_avg,
            perceived_ease_of_use=peou_avg,
            behavioral_intention=bi,
            adoption_risk=risk,
            recommendations=recs,
        )
