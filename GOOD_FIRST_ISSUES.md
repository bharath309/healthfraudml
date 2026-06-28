# Good First Issues — Ready to Create on GitHub

Create these as GitHub Issues with the label `good first issue`. These attract new contributors, which strengthens the project's community evidence for EB-1A.

---

## Issue 1: Add unit tests for BillingAuditor fraud detection rules

**Labels**: `good first issue`, `testing`

The `BillingAuditor` class has 6 fraud detection methods but no unit tests. Add pytest tests covering:
- `check_upcoding()` with claims above and below threshold
- `check_unbundling()` with bundled and unbundled procedure sets
- `check_phantom_billing()` with valid and phantom provider/patient combos
- Edge cases: empty dataframes, single-row inputs, missing columns

**Files**: `healthfraudml/auditor/billing_auditor.py` → create `tests/test_billing_auditor.py`

---

## Issue 2: Add type hints throughout the codebase

**Labels**: `good first issue`, `code quality`

Add Python type hints (PEP 484) to all public methods in:
- `healthfraudml/auditor/billing_auditor.py`
- `healthfraudml/auditor/llm_integration.py`
- `experiments/run_comparison.py`

Example: `def audit_claims(self, claims: pd.DataFrame) -> Dict[str, Any]:`

---

## Issue 3: Add CSV/JSON input loader utility

**Labels**: `good first issue`, `enhancement`

Create a `healthfraudml/utils/data_loader.py` that:
- Loads claims from CSV, JSON, or Parquet files
- Validates required columns exist (cpt_code, amount, provider_id, patient_id)
- Returns a standardized DataFrame with consistent column names
- Provides helpful error messages for malformed data

---

## Issue 4: Create interactive Jupyter notebook tutorial

**Labels**: `good first issue`, `documentation`

Create `notebooks/quickstart.ipynb` that walks through:
1. Loading sample data
2. Running the BillingAuditor
3. Interpreting results
4. Visualizing fraud patterns with matplotlib

---

## Issue 5: Add pre-commit hooks configuration

**Labels**: `good first issue`, `infrastructure`

Add a `.pre-commit-config.yaml` with:
- black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

Include setup instructions in CONTRIBUTING.md.

---

## Issue 6: Support ICD-10 code validation

**Labels**: `good first issue`, `enhancement`

Add ICD-10 diagnosis code validation to the auditor:
- Validate ICD-10 format (letter + digits pattern)
- Flag impossible age/diagnosis combinations (e.g., pediatric diagnoses for elderly patients)
- Cross-reference CPT-ICD compatibility

---

## GitHub Topics to Add

Go to repo Settings → Topics and add these 10 tags:

```
healthcare-fraud-detection
machine-learning
fraud-detection
healthcare-analytics
python
scikit-learn
explainable-ai
billing-audit
medical-claims
open-source
```
