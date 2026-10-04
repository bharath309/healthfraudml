# CMS–LEIE arm — frozen protocol (gate 2), written 3 Oct 2026 before the first model was trained

Companion to `ClaudeCode_Brief_CMS_LEIE_Arm_Oct3.md` (§3) and `label_report.md` (gate 1). Deviations after this point are
appended with a date and reason, never edited in place. Scripts: `download.py`, `build_labels.py`, `build_features.py`,
`make_folds.py`, `cmsarm.py`, `resample_fold.py`, `train_model.py`, `summarise.py` (this folder).

## Rows
All 13,527,801 kept US provider-years, calendar years **2013–2024** (CY2024 was published 21 May 2026 and is included,
as the brief instructed). Primary label `is_fraud` (authorities 1128(a)(1), (a)(2), (a)(3), (b)(4), (b)(7); the two
(c)(3)(G) codes are absent from the 10 Sep 2026 LEIE); 11,169 positives (0.0826%). Year is not a feature.

## Features (primary set, 55 columns, `out/feature_list.json` in this order)
- 37 published numerics: `Tot_HCPCS_Cds, Tot_Benes, Tot_Srvcs, Tot_Sbmtd_Chrg, Tot_Mdcr_Alowd_Amt, Tot_Mdcr_Pymt_Amt,
  Tot_Mdcr_Stdzd_Amt`, the seven `Drug_*` totals, the seven `Med_*` totals, `Bene_Avg_Age`, four age-band counts, two sex
  counts, six race counts, `Bene_Dual_Cnt`, `Bene_Ndual_Cnt`, `Bene_Avg_Risk_Scre`.
- 11 derived, zero-safe: `srvcs_per_bene, chrg_per_srvc, alowd_per_srvc, pymt_over_alowd, stdzd_over_pymt, drug_share_pymt,
  hcpcs_per_1000_srvcs, dual_share, female_share, share_65_74, share_85p`.
- 2 indicators: `block_suppressed` (the Drug/Med suppression flags coincide row-wise, so one column); `bene_demo_n_missing`
  (number of suppressed demographic cells, 0–15 — the brief's "any demographic missing" indicator was replaced at gate 2
  because it is set on ~97% of rows and carries no information; `risk_missing` was dropped because the risk score is never
  blank). Blank numerics are 0.
- 5 label-encoded categoricals (sorted-vocabulary codes over all years, `out/category_vocab.json`): provider type, state,
  entity code, RUCA first digit, participation indicator. (No gender field exists in the released files.)
- Scaling: RobustScaler convention — (x − median)/IQR with IQR 0 → 1 — statistics over all kept rows of all years
  (`out/scaler_stats.json`), applied identically to training and test rows (inherited convention from `run_comparison`).
- Sensitivity feature set `sensitivity_cc_2017plus` (80 columns): the 55 above + `cc_missing` + the 25 chronic-condition
  percentages, rows 2017–2024 only.

## Splits and sampling
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, y = `is_fraud`, groups = NPI; provider-disjointness
  asserted; assignment saved in `data/folds.parquet` (`out/folds_report.json`: test folds 2,705,560–2,705,561 rows,
  2,233–2,234 positives each).
- Training set for fold k: all 8,935–8,936 positives of the four training folds + 178,700–178,720 negatives sampled at
  **20 per positive** with `numpy.random.default_rng(42 + k)` (`data/train_<k>.parquet`), then
  `SMOTEENN(random_state=42)` exactly as the pipeline; the resampled matrix is cached once per fold
  (`data/resampled_primary_is_fraud_<k>.npz`) so every model trains on identical rows. Fold 1: 187,635 → 310,840 rows
  (172,993 positive) in 362 s.
- Test fold: complete, natural prevalence.

## Models (hyper-parameters identical to `run_comparison.run_experiments`, `random_state=42`)
Random Forest (500, depth 20, min_samples_leaf 5, class_weight balanced, n_jobs −1); SVM (RBF, C 10, gamma scale,
class_weight balanced, probability True); Neural Network MLP (128-64-32, relu, max_iter 200, early_stopping,
validation_fraction 0.1); Gradient Boosting (500, depth 6, lr 0.05, subsample 0.8); Gaussian Naive Bayes; soft-voting
ensemble of RF (200, depth 15, balanced) + GB (200, depth 5, lr 0.05) + SVC (RBF, C 10, balanced, probability True).

Compute mechanics fixed now (they do not change any fitted model):
- GB and RF are trained in warm-start increments with the estimator pickled between increments; verified on 3 Oct that
  the warm-started estimators give probabilities identical (max |Δ| = 0.0) to a single `fit` with the same seed.
- The two SVC-based models (SVM; the ensemble's SVC) are fitted on a seeded 30,000-row subsample of the resampled training
  fold, `default_rng(1000 + k)`; everything else on the whole resampled fold.
- The soft-voting ensemble is computed as the unweighted mean of the three members' `predict_proba`, which is what
  `VotingClassifier(voting="soft")` returns; each member is fitted with `random_state=42` as the pipeline clones them.
- Hardware: Cowork cloud workspace, 2 CPUs, 7 GB RAM (GB ≈ 60 min per fold); the Mac re-run via Claude Code must give the
  same numbers from the same files (SHA-256) and seeds.

## Metrics and outputs
Per model per fold (`results/<model>/fold<k>.json`): F1, precision, recall, MCC at `predict()` (0.5); AUC-PR
(`average_precision_score`); AUC-ROC; confusion matrix; sizes; seconds; versions. OOF probabilities for every model:
`results/oof_fold<k>.parquet` (npi, year, is_fraud, p_<model>). `results/summary.json`: mean ± SD and rank orders by F1
and AUC-PR. Threshold study for Gradient Boosting from the pooled OOF probabilities: 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95;
grid optimum (0.01 steps); nested selection. Everything written as soon as it exists.

## Sensitivities (after the primary run; GB and RF unless stated)
1. `is_fraud_mandatory` (all six models). 2. `is_fraud_5yr`. 3. Year as a feature. 4. Negatives at 50 per positive.
5. (added at gate 2) Feature set `sensitivity_cc_2017plus`, rows 2017–2024.

## Order of execution
Primary: resample each fold → GB, RF, NB, MLP, SVM, ensemble members → summaries → threshold study → sensitivities.
Results are reported to Bharath only after gate-3 checks (`Verification_Checklist_CMS_LEIE_Arm_Oct3.md`).

## Appended 3 Oct 2026, 13:20 UTC — execution notes (no change to any fitted model)
- Primary run completed in the cloud workspace for Gradient Boosting, Random Forest, MLP and Naive Bayes on all five folds
  (`results/primary_is_fraud/`); the ensemble members RF(200) are complete and GB(200) is running; SVM and the ensemble's
  SVC are not complete here: the 30,000-row SVC fit took longer than the 10-minute foreground limit and scoring 2.7 M test
  rows took ~16 min per 250,000-row batch, so these two models (fold 1 is in progress in the background; its fitted state
  and partial predictions are saved) and the sensitivities are assigned to the Mac re-run described in `RUN_ON_MAC.md`,
  which is also the independent verification of everything computed here.
- Chunking: GB and GB(200) were trained in 30-tree warm-start increments (fit time per fold 3,860–4,082 s); RF in 100-tree
  increments; predictions in 250,000-row batches. None of this changes a fitted model (verified 3 Oct, see above).
- The per-fold JSON metrics were recomputed from the stored probabilities by `summarise.py`: maximum absolute difference 0.0
  for every model and fold. Provider-disjointness re-checked on `folds.parquet`: 0 providers in more than one fold.
- Sensitivity label sets reuse `train_<k>.parquet` (sampled under the primary label); rows that are primary-positive but not
  mandatory-positive (or outside the 5-year window) are negatives under those labels, so the effective negative:positive
  ratio there is higher than 20:1 — report it with the sensitivity results.

## Appended 3 Oct 2026, 17:35 UTC — SVC scoring path and the ensemble's SVC member (no change to any fitted model)
- SVC models are scored by `fastsvc.py`: the RBF kernel against the support vectors computed with BLAS, the decision value
  `K·dual_coef + intercept`, and the probability exactly as scikit-learn's bundled libsvm computes it (Platt sigmoid of the
  sign-flipped decision value, clipped to [1e-7, 1−1e-7], then the Wu–Lin–Weng `multiclass_probability` iteration, which
  libsvm runs even for two classes). Validated on 5,000 test rows against `SVC.decision_function`/`predict_proba`:
  max |Δ| 2.6e-9 and 3.3e-10. Fold 1 of the SVM carries its first 250,000 rows from libsvm itself and the rest from this
  path (float32 storage differences ≤ 1.5e-7).
- The ensemble's SVC member is, by the pipeline's own definition, the same estimator as the standalone SVM (same
  hyper-parameters, seed and 30,000-row subsample); its fold-1 run reproduced the SVM's predictions to float32 rounding
  (max |Δ| 6e-8), so folds 2–5 of `svc200` reuse the SVM's stored predictions (noted in each `svc200/fold<k>.json`).
- Ensemble probability = mean of `rf200`, `gb200`, `svc200` probabilities (`summarise.py`), as `VotingClassifier(voting="soft")`.
- Memory: the cloud workspace's cgroup kills a process above ~3 GB RSS; SVC kernel chunks are 1,000 rows for that reason.
