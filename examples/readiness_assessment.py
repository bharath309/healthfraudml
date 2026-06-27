"""
Example: Using the Organizational Readiness Assessment Tool.

Demonstrates how healthcare administrators can evaluate their
institution's readiness to adopt ML for fraud detection.
"""

from healthfraudml.readiness import ReadinessAssessment


def main():
    assessment = ReadinessAssessment()

    # Example: Small community hospital
    print("=" * 60)
    print("READINESS ASSESSMENT: Small Community Hospital")
    print("=" * 60)

    report = assessment.evaluate(
        institution_size="small",
        annual_claims_volume=30000,
        existing_fraud_detection="manual",
        it_staff_count=2,
        annual_fraud_budget=15000,
        has_data_science_team=False,
        data_digitized=True,
        has_labeled_fraud_data=False,
        leadership_support=True,
        staff_tech_comfort=2,
        regulatory_requirements=True,
    )

    print(report.summary())
    print("\nDimension Scores:")
    for dim, score in report.dimension_scores.items():
        print(f"  {dim.replace('_', ' ').title()}: {score:.1f}/100")

    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\nImplementation Roadmap:")
    for phase in report.implementation_roadmap:
        print(f"  Phase {phase['phase']} (Months {phase['months']}): {phase['focus']}")
        for action in phase["actions"]:
            print(f"    - {action}")

    # Example: Large hospital system
    print("\n" + "=" * 60)
    print("READINESS ASSESSMENT: Large Hospital System")
    print("=" * 60)

    report2 = assessment.evaluate(
        institution_size="large",
        annual_claims_volume=500000,
        existing_fraud_detection="rule_based",
        it_staff_count=25,
        annual_fraud_budget=200000,
        has_data_science_team=True,
        data_digitized=True,
        has_labeled_fraud_data=True,
        leadership_support=True,
        staff_tech_comfort=4,
        regulatory_requirements=True,
    )

    print(report2.summary())


if __name__ == "__main__":
    main()
