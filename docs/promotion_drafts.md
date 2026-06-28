# HealthFraudML — Promotional Copy Drafts

This document contains pre-written, highly optimized text copy for sharing and promoting **HealthFraudML** on various online communities to gain visibility and stars.

---

## 1. LinkedIn Announcement

**Target Audience**: Healthcare executives, ML Engineers, Clinical Researchers, and Health IT specialists.
**Tone**: Professional, academic, impact-driven.

***

**Copy**:

> I am thrilled to share the initial open-source release of **HealthFraudML**, a modular Python framework designed for detecting financial fraud in healthcare billing claims using Machine Learning.
> 
> 🔗 **GitHub Repository**: https://github.com/bharath309/healthfraudml
> 🔗 **Jupyter Demo (Google Colab)**: https://colab.research.google.com/github/bharath309/healthfraudml/blob/main/examples/healthfraudml_demo.ipynb
> 
> Healthcare billing fraud costs the U.S. system billions annually. While ML algorithms show promise in research settings, real-world adoption is blocked by organizational and data silo barriers.
> 
> Grounded in my doctoral dissertation at the University of the Cumberlands (*"Financial Fraud Detection in Healthcare Settings"*), this framework aims to bridge the gap between academic ML theory and clinic-level workflow reality.
> 
> ### Key Features:
> ✅ **Preprocessing Pipelines**: Configured for ICD-10, CPT, and NDC billing codes.
> ✅ **8+ Algorithms**: Inbuilt supervised, unsupervised, and hybrid ensembles (e.g., AdaBoost + Voting) designed for handling high class imbalance.
> ✅ **Organizational Readiness Tool**: An assessment checklist grounded in the Technology Acceptance Model (TAM) and Diffusion of Innovations (DOI) to score an institution's adoption readiness (0-100).
> ✅ **Explainable AI (XAI)**: Flagged claims are output with human-readable explanations to help auditors understand the classification decision.
> 
> If you work in healthcare IT, medical audit compliance, or financial machine learning, I would love for you to check it out, run the Google Colab quickstart, and share your feedback! 
> 
> Let's make healthcare claims processing more transparent and fraud-resistant.
> 
> \#MachineLearning \#HealthcareIT \#HealthTech \#OpenSource \#Python \#DataScience \#FraudDetection \#AIInClinicalMedicine

---

## 2. Reddit Posts

### Post A: For `r/MachineLearning` & `r/datascience`
**Tone**: Technical, focused on design choices, models, and class imbalance.

*   **Title**: `[Project] HealthFraudML – An Open-Source Python Framework for Healthcare Claims Fraud Detection & Explainability`
*   **Body**:
    > Hi everyone,
    > 
    > I just released the first version of **HealthFraudML**, a Python package designed specifically to build, evaluate, and benchmark ML fraud detectors on healthcare claims data.
    > 
    > *   **Code**: https://github.com/bharath309/healthfraudml
    > *   **Quickstart Demo**: https://colab.research.google.com/github/bharath309/healthfraudml/blob/main/examples/healthfraudml_demo.ipynb
    > 
    > ### Why this package?
    > Most fraud detection packages are built for general credit card fraud. Healthcare claims have highly unique problems:
    > 1. **Categorical Complexity**: Claims rely on large hierarchies of ICD diagnostic, CPT procedural, and NDC drug codes.
    > 2. **Severe Class Imbalance**: Fraud rates in real claims are often less than 1%.
    > 3. **Black Box Resistance**: Medical billing auditors reject model predictions if they cannot see *why* a claim was flagged.
    > 
    > ### Technical Highlights:
    > *   **Unified API**: `FraudDetector` wrapper that integrates supervised models (Neural Nets, Gradient Boosting, SVMs), unsupervised models (K-Means/X-Means, AIS), and hybrid ensembles.
    > *   **Imbalance Handling**: Preprocessing pipeline handles code encoding, scaling, and class weights.
    > *   **XAI Output**: Integrated explainability that generates natural-language reasons for flagged billing claims.
    > *   **Readiness Evaluator**: An assessment module based on the Technology Acceptance Model (TAM) and Diffusion of Innovations (DOI) to evaluate organizational readiness.
    > 
    > Feel free to star the repo, run the Colab, or contribute!

### Post B: For `r/HealthIT` & `r/medicine`
**Tone**: Applied, operational, auditing, compliance.

*   **Title**: `HealthFraudML: Open-Source Machine Learning Tool for Healthcare Claims Fraud Auditing`
*   **Body**:
    > Hello Health IT community,
    > 
    > I have released **HealthFraudML**, an open-source Python framework designed to help healthcare systems audit billing transactions and detect fraud (upcoding, phantom billing, duplicate claims, unbundling).
    > 
    > *   **GitHub**: https://github.com/bharath309/healthfraudml
    > 
    > Grounded in research on ML adoption across U.S. hospitals, the tool includes:
    > 
    > 1. **Automated Auditing Rules**: Preconfigured detectors matching billing codes for anomalies.
    > 2. **Adoption Readiness Assessment**: A survey module for administrators to evaluate if their IT infrastructure and staff are ready to integrate AI tools (calculating a 0-100 score).
    > 3. **Auditor Explanations**: Instead of just outputting a probability, it explains *why* a claim was flagged in human-readable terms to support manual reviews.
    > 
    > The goal is to provide a free, developer-friendly toolkit that hospital systems can host locally to evaluate fraud risk without committing to multi-million dollar vendor software.

---

## 3. Hacker News (Show HN)

*   **Title**: `Show HN: HealthFraudML – ML framework for healthcare billing fraud detection`
*   **Body**:
    > HealthFraudML is a Python library targeting financial fraud in healthcare claims (CPT/ICD codes).
    > 
    > Repository: https://github.com/bharath309/healthfraudml
    > Colab Demo: https://colab.research.google.com/github/bharath309/healthfraudml/blob/main/examples/healthfraudml_demo.ipynb
    > 
    > Existing fraud frameworks are heavily geared toward credit card transactions. Medical fraud (e.g., upcoding, phantom billing, duplicate claims) is different because it relies on hierarchical nomenclature (ICD, CPT, NDC) and requires strict explainability for billing auditors.
    > 
    > This framework packages:
    > - Claims preprocessing pipelines with category encoding
    > - Supervised & unsupervised models (including Artificial Immune Systems)
    > - Hybrid ensemble models (AdaBoost + Voting)
    > - An institutional readiness evaluator based on TAM/DOI models to assess IT and staff capabilities
    > 
    > The project was built out of my Ph.D. research at the University of the Cumberlands on ML adoption barriers. I'd love your feedback on the API design and the hybrid model structure.

---

## 4. Medium / Dev.to Technical Article Outline

**Proposed Title**: `Building an Open-Source Healthcare Fraud Detection Framework with Machine Learning`

*   **Introduction**: The immense cost of medical billing fraud and the limitations of rule-based firewalls.
*   **The Problem with Off-the-Shelf ML**: Why generic tabular model setups fail on hierarchical medical codes (CPT/ICD) and the necessity of strict HIPAA compliance.
*   **HealthFraudML Architecture**:
    *   *Preprocessing*: Dealing with extreme class imbalance (under 5% target class) and categorical hierarchies.
    *   *Model Zoo*: Comparison of neural nets, Random Forests, and hybrid models.
    *   *Explainable Predictions*: Why providing a probability score is not enough for medical coders, and how we build XAI text outputs.
*   **The Human Element**: Implementing a programmatic TAM/DOI readiness survey to ensure organizations have the IT infrastructure to maintain the models.
*   **Conclusion & Next Steps**: Invitation to join the open-source community on GitHub.
