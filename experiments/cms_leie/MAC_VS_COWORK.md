# Mac verification run vs Cowork cloud run — primary results (4 Oct 2026, 03:20 UTC)

**Verdict: the four-decimal agreement that RUN_ON_MAC.md expected did not hold for 5 of 6 models. Every conclusion
holds: the same model ranking on all four metrics, and mean differences far smaller than the spread across folds.
Cause: floating-point differences between machines, not the code or the data.**

## What matched exactly
- All 13 input files: SHA-256 and byte counts identical (`mac_compare_downloads.md`), including the LEIE 10 Sep 2026 snapshot.
- `label_report.json`, `folds_report.json`, `category_vocab.json`, `scaler_stats.json`, `feature_list.json` (after
  `add_sensitivity_stats.py`; see below) — `mac_compare_reports.md`.
- SMOTE-ENN output sizes: rows and positives after resampling are identical on all five folds.
- Library versions identical (scikit-learn 1.8.0, imbalanced-learn 0.14.2, numpy 2.4.4, pandas 3.0.2, pyarrow 25.0.1,
  scipy 1.17.1). Python 3.11.7 (Mac) vs 3.11.15 (Cowork).
- Naive Bayes: every metric and confusion count identical on all five folds.

## What differed (`mac_compare_primary.md`)
Per-fold max |metric diff|: GB 2.2e-3 to 5.8e-3; RF 4.5e-4 to 2.2e-3; MLP 4.0e-3 to 2.5e-2; SVM 1.4e-4 to 5.8e-4;
ensemble 9.0e-4 to 4.0e-3. Confusion counts differ by a handful of cases per fold (e.g. GB fold 1: TP 652 vs 647).
Model internals show the same pattern: MLP stopped after 106 iterations vs 85 (fold 1); SVM has 12,771 vs 12,770
support vectors.

| model | F1 Mac / Cowork | AUC-PR Mac / Cowork | AUC-ROC Mac / Cowork | MCC Mac / Cowork |
|---|---|---|---|---|
| Gradient Boosting | 0.0175 / 0.0177 | 0.0102 / 0.0101 | 0.8009 / 0.8014 | 0.0476 / 0.0481 |
| Random Forest | 0.0108 / 0.0108 | 0.0082 / 0.0083 | 0.7991 / 0.7992 | 0.0407 / 0.0406 |
| MLP | 0.0030 / 0.0030 | 0.0019 / 0.0019 | 0.6735 / 0.6694 | 0.0154 / 0.0151 |
| Naive Bayes | 0.0025 / 0.0025 | 0.0013 / 0.0013 | 0.5859 / 0.5859 | 0.0009 / 0.0009 |
| SVM (RBF) | 0.0029 / 0.0029 | 0.0012 / 0.0012 | 0.6161 / 0.6161 | 0.0045 / 0.0045 |
| Ensemble | 0.0081 / 0.0080 | 0.0080 / 0.0080 | 0.7948 / 0.7947 | 0.0362 / 0.0360 |

- Largest difference in a mean: MLP AUC-ROC +0.0041, about one third of its fold-to-fold SD (0.0125).
- Ranking by F1, AUC-PR, AUC-ROC and MCC: identical to Cowork on all four.
- GB threshold study: grid optimum 0.89 (Mac) vs 0.88 (Cowork); nested F1 0.0445 ± 0.0046 vs 0.0457 ± 0.0056.
  If the manuscript quotes the optimal threshold, quote it as about 0.88–0.89, not to two decimals.

## Why — evidence
- Deterministic within a machine: MLP fold 1 and RF fold 1 re-trained from scratch on the Mac reproduce the first Mac
  run exactly (every metric, every count, MLP iteration count).
- Different hardware and maths libraries: the Mac is Apple Silicon (arm64, numpy linked to Apple Accelerate). Cowork
  ran on a 2-CPU Linux cloud workspace. Rounding in the last bits of floating-point sums and distances differs between
  these, and it compounds through iterative training (MLP epochs, boosting rounds, SVM optimisation, near-tie tree
  splits). Naive Bayes is a single pass of means and variances, so it does not compound — consistent with its exact match.
- Not proven: whether the SMOTE-ENN synthetic rows are bit-identical across machines (sizes are identical; the Cowork
  `.npz` files are not on the Mac). The conclusion is the same either way.

## Implication for the manuscript (Cowork gate 3 decides)
- Report results as reproducible to about 3 decimals across platforms, and exactly within one platform. Name the
  platform the reported numbers come from.
- "Expect four-decimal agreement" (RUN_ON_MAC.md, protocol_frozen.md hardware note) should be amended for cross-platform reruns.

## Other notes
- `add_sensitivity_stats.py` (new): writes the `cc_missing` and `year` scaler statistics and the `primary_plus_year`
  feature set, which Cowork had added to the JSON by hand. The Mac values equal Cowork's exactly.
- `cc_missing` is 1 in at least 75% of 2017–2024 rows (q25 = median = q75 = 1). Look at this before interpreting the
  chronic-condition sensitivity.
- `cowork_reference/` holds Cowork's outputs as copied on 3 Oct; it contains no sensitivity results, so the Mac
  sensitivities have no counterpart to compare with here.

## Sensitivities on the Mac (GB and RF, 5 folds each; finished 4 Oct 2026, 15:09 UTC)
Run with `mac_run_sens_parallel.sh` (same commands as `run_sensitivity.sh`, one job per sensitivity × fold). No Cowork
sensitivity results were in `cowork_reference/`, so there is nothing to compare against here; Cowork performs gate 3.
`folds_report_r50.json` is identical to Cowork's. Mean ± SD over folds:

| Run | GB F1 | GB AUC-PR | GB AUC-ROC | RF F1 | RF AUC-PR | RF AUC-ROC |
|---|---|---|---|---|---|---|
| Primary (reference) | 0.0175 ± 0.0027 | 0.0102 ± 0.0012 | 0.8009 ± 0.0155 | 0.0108 ± 0.0009 | 0.0082 ± 0.0016 | 0.7991 ± 0.0181 |
| Mandatory exclusions only | 0.0183 ± 0.0022 | 0.0143 ± 0.0032 | 0.8432 ± 0.0196 | 0.0105 ± 0.0010 | 0.0092 ± 0.0023 | 0.8466 ± 0.0185 |
| 5-year label window | 0.0120 ± 0.0013 | 0.0029 ± 0.0005 | 0.7858 ± 0.0156 | 0.0066 ± 0.0010 | 0.0028 ± 0.0008 | 0.7946 ± 0.0183 |
| + chronic conditions, 2017–2024 | 0.0169 ± 0.0036 | 0.0061 ± 0.0013 | 0.7872 ± 0.0243 | 0.0096 ± 0.0017 | 0.0043 ± 0.0010 | 0.7869 ± 0.0224 |
| + year as feature | 0.0245 ± 0.0034 | 0.0126 ± 0.0019 | 0.8306 ± 0.0103 | 0.0107 ± 0.0007 | 0.0093 ± 0.0017 | 0.8182 ± 0.0118 |
| 50:1 negatives | 0.0262 ± 0.0044 | 0.0091 ± 0.0014 | 0.7890 ± 0.0157 | 0.0150 ± 0.0016 | 0.0083 ± 0.0016 | 0.7933 ± 0.0171 |

- GB beats RF on F1 and AUC-PR in all six runs. RF is marginally higher on AUC-ROC in three (mandatory, 5-year,
  50:1), each gap smaller than one fold SD.
- Mandatory-exclusion labels give the highest AUC-ROC (0.84–0.85); the 5-year window gives the lowest AUC-PR (0.003).
- Year as a feature raises GB F1 from 0.0175 to 0.0245, which suggests time trends carry signal; worth a look in gate 3.
- Same cross-platform caveat as the primary results: expect agreement with any Cowork rerun to about 3 decimals.
