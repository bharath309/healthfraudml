"""Generate readiness assessment reports in various formats."""

from healthfraudml.readiness.assessment import ReadinessReport
from typing import Optional


def generate_text_report(report: ReadinessReport) -> str:
    """Generate a plain-text readiness report."""
    lines = [
        "=" * 60,
        "HEALTHCARE FRAUD DETECTION: ML READINESS ASSESSMENT REPORT",
        "=" * 60,
        "",
        f"Overall Readiness Score: {report.readiness_score:.1f} / 100",
        f"Readiness Level: {report.readiness_level}",
        "",
        "DIMENSION SCORES",
        "-" * 40,
    ]

    for dim, score in report.dimension_scores.items():
        bar = "#" * int(score / 5) + "." * (20 - int(score / 5))
        lines.append(f"  {dim.replace('_', ' ').title():<35} [{bar}] {score:.1f}")

    lines.extend([
        "",
        f"Strongest Dimension: {report.strongest_dimension.replace('_', ' ').title()}",
        f"Weakest Dimension: {report.weakest_dimension.replace('_', ' ').title()}",
        "",
        "RECOMMENDATIONS",
        "-" * 40,
    ])

    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    lines.extend(["", "IMPLEMENTATION ROADMAP", "-" * 40])

    for phase in report.implementation_roadmap:
        lines.append(f"\n  Phase {phase['phase']}: {phase['focus']} (Months {phase['months']})")
        for action in phase["actions"]:
            lines.append(f"    - {action}")

    lines.extend([
        "",
        "=" * 60,
        "Based on: Bahudhoddi, B.K. (2025). Financial Fraud Detection",
        "in Healthcare Settings. University of the Cumberlands.",
        "=" * 60,
    ])

    return "\n".join(lines)
