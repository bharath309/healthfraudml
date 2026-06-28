# Contributing to HealthFraudML

Thank you for your interest in contributing to HealthFraudML! This project aims to make healthcare fraud detection more accessible through open-source ML tools, and contributions from the community are essential to that mission.

## How to Contribute

### Reporting Bugs

- Open a [GitHub Issue](https://github.com/bharath309/healthfraudml/issues) with the label `bug`
- Include: Python version, OS, steps to reproduce, expected vs. actual behavior, and full traceback

### Suggesting Features

- Open an issue with the label `enhancement`
- Describe the use case and how it fits into healthcare fraud detection workflows

### Code Contributions

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature-name`
3. **Install in development mode**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/healthfraudml.git
   cd healthfraudml
   pip install -e ".[dev]"
   ```
4. **Make your changes** — follow the code style guidelines below
5. **Add or update tests** in `tests/`
6. **Run tests**: `pytest tests/ -v`
7. **Commit** with a clear message: `git commit -m "Add: description of change"`
8. **Push** and open a Pull Request against `main`

### Good First Issues

New to the project? Look for issues labeled [`good first issue`](https://github.com/bharath309/healthfraudml/labels/good%20first%20issue). These are scoped, well-defined tasks ideal for first-time contributors:

- Add new CPT codes to the reference database (e.g., radiology, pathology codes)
- Improve regex patterns in the billing parser fallback
- Add unit tests for edge cases (empty inputs, malformed PDFs, missing fields)
- Expand provider name detection patterns beyond the current 5
- Add new fraud detection heuristics (e.g., phantom billing, duplicate claims)
- Improve documentation or fix typos

## Code Style

- **Python 3.8+** compatibility
- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for all function signatures
- Write docstrings in NumPy format
- Keep functions focused — one responsibility per function

## Testing

- All new features must include tests
- Tests go in `tests/` and use `pytest`
- Aim for meaningful assertions, not just "it runs without error"
- Run the full suite before submitting: `pytest tests/ -v`

## Data & Privacy

- **Never commit real patient data** — use synthetic or anonymized data only
- All example data must be clearly labeled as synthetic
- Follow HIPAA de-identification guidelines when creating test fixtures

## Project Structure

```
healthfraudml/
├── detector/          # Core fraud detection models
├── models/            # ML algorithm implementations
│   ├── supervised/    # Random Forest, SVM, Neural Net, etc.
│   ├── unsupervised/  # K-Means, Outlier Detection, AIS
│   └── hybrid/        # Ensemble methods
├── preprocessing/     # Claims data pipelines
├── readiness/         # Organizational readiness assessment
├── evaluation/        # Benchmarking and metrics
├── auditor/           # Patient billing auditor
│   ├── billing_auditor.py   # Rule-based CPT audit engine
│   ├── llm_integration.py   # Gemini LLM parser + RAG auditor
│   └── db.py                # ChromaDB vector store for CPT rules
├── privacy/           # HIPAA-compliant utilities
└── utils/             # Shared helpers
```

## Questions?

Open a [Discussion](https://github.com/bharath309/healthfraudml/discussions) or reach out to the maintainer.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
