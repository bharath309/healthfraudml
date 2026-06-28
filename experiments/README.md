# Running the Experiment

## Prerequisites

```bash
pip install numpy pandas scikit-learn imbalanced-learn shap
```

## Quick Start (Synthetic Data Only)

```bash
cd experiments
python run_comparison.py
```

## Using Real Kaggle Data

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
2. Place the CSV files in `data/kaggle/`:
   ```
   data/kaggle/
   ├── Train_Beneficiarydata.csv
   ├── Train_Inpatientdata.csv
   ├── Train_Outpatientdata.csv
   └── Train.csv
   ```
3. Run:
   ```bash
   python run_comparison.py --kaggle           # Kaggle only
   python run_comparison.py --both             # Both datasets
   python run_comparison.py --data-dir /path/  # Custom data location
   ```

## Output

Results are saved to `../results/<dataset_name>/`:

- `synthetic_claims.csv` — 50K synthetic claims dataset (3.2% fraud rate)
- `experiment_results.json` — Full metrics for all 8 models
- `results_table.md` — Formatted comparison table for the paper

## What It Does

1. Loads data — either synthetic (50K claims, 5 fraud types) or Kaggle Medicare provider-level claims
2. Engineers features — provider profiling, claim statistics, specialty deviation, chronic condition counts
3. Trains 6 supervised/hybrid models + K-Means + AIS via 5-fold stratified CV with SMOTE-ENN resampling
4. Evaluates using F1-score, AUC-PR, and MCC (not accuracy — meaningless at 96%+ non-fraud)
5. Runs SHAP feature importance analysis on Gradient Boosting
6. Exports results as JSON and markdown table

## After Running

Copy the numbers from `results/*/results_table.md` into Paper 2, replacing the placeholder values in Tables 1-3. Using both datasets strengthens the paper — synthetic shows controllability, Kaggle shows real-world applicability.
