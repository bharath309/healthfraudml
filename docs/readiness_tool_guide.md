# Organizational Readiness Assessment Guide

A major barrier to ML adoption in clinical fraud detection is the lack of institutional readiness. **HealthFraudML** implements a scoring framework based on three models: the **Technology Acceptance Model (TAM)**, **Fraud Triangle Theory**, and **Diffusion of Innovations Theory (DOI)**.

## Usage

Healthcare administrators can calculate their readiness score and generate recommendations:

```python
from healthfraudml.readiness import ReadinessAssessment

assessment = ReadinessAssessment()
report = assessment.evaluate(
    institution_size="medium",      # "small", "medium", "large"
    annual_claims_volume=100000,
    existing_fraud_detection="manual",
    it_staff_count=4,
    annual_fraud_budget=30000,
    has_data_science_team=False,
    data_digitized=True,
    has_labeled_fraud_data=True,
    leadership_support=True,
    staff_tech_comfort=3,
    regulatory_requirements=True
)

print(f"Readiness Score: {report.readiness_score}/100")
print(f"Readiness Level: {report.readiness_level}")
```

## Assessment Dimensions

1.  **Organizational Infrastructure (25%)**: IT bandwidth, hardware resources, and historical audit workflows.
2.  **Data Readiness (20%)**: Claims structure, historic volume, labeling quality, and HIPAA privacy rules.
3.  **Leadership & Culture (20%)**: Executive buy-in, budget allocations, and change receptiveness.
4.  **Staff Capability & Training (20%)**: Analytical background and technical comfort.
5.  **Regulatory Environment (15%)**: Audit compliance pressure and regulatory penalties.
