# Changelog

All notable changes to HealthFraudML will be documented in this file.

## [0.2.0] - 2026-07-17

### Added
- **CMS PFS price benchmark**: price benchmarking expanded from 10 curated codes to
  7,359 separately payable CPT/HCPCS codes, generated from the public CMS Physician
  Fee Schedule Relative Value File (2025). Ships as code numbers + national payment
  ranges only — no AMA CPT descriptions (see `healthfraudml/auditor/data/README.md`
  for provenance and the disclosed `fair_max = medicare_max × 5.0` heuristic).
- **`scripts/build_cms_benchmark.py`**: regenerates the benchmark from any CMS PPRRVU
  release (`--cf`, `--fair-mult` tunable).
- **`[rag]` extra**: `pip install "healthfraudml[rag]"` installs the optional
  ChromaDB + Google GenAI + pypdf dependencies for the RAG/LLM modules.
- **Pilot quickstart**: `examples/pilot_audit.py` (CSV in → flagged-claims report +
  draft dispute letter out, fully offline) + `examples/sample_claims_pilot.csv`.

### Changed
- Curated 10-code reference now overlays the CMS benchmark: expanded (price-only)
  codes can raise overpricing findings but never upcoding — severity stays exclusive
  to curated codes by design.
- README/Colab install path switched to `pip install healthfraudml` (live on PyPI).
- numpy/pandas capped `<3` to avoid pandas 3.0 breakage.

### Notes
- The forward-looking items listed under "What's Next" in the v0.1.0 release notes
  (guideline-retrieval RAG, streaming pipeline, pre-trained weights, dashboard)
  remain roadmap candidates and are NOT part of this release.

## [0.1.0] - 2026-06-27

### Added
- **Core Framework**: Healthcare claims fraud detection pipeline with modular architecture
- **BillingAuditor**: Rule-based billing audit engine covering 6 fraud patterns (upcoding, unbundling, phantom billing, duplicate claims, identity fraud, anomalous patterns)
- **LLM Integration**: Optional GPT/Claude integration for natural language fraud explanations
- **ML Comparison Suite**: Experiment script comparing 8 ML architectures:
  - Random Forest, SVM (RBF), Neural Network (MLP), Gradient Boosting, Naive Bayes, K-Means Clustering, Artificial Immune System (Negative Selection), AdaBoost-Voting Ensemble
- **Synthetic Data Generator**: Configurable healthcare claims dataset with realistic fraud distributions (50K claims, 3.2% fraud rate, 20 CPT codes, 500 providers)
- **SMOTE-ENN Resampling**: Hybrid oversampling/cleaning for severe class imbalance
- **Explainability**: SHAP-based feature importance analysis with fallback to Gini importance
- **Evaluation Metrics**: F1-score, AUC-PR, MCC (fraud-appropriate metrics, not accuracy)
- **Documentation**: Comprehensive README, contributing guidelines, and experiment instructions

### Infrastructure
- Python package structure with `setup.py` and `pip install -e .` support
- MIT License
- GitHub Issue templates for bugs and feature requests
