# Protocol T — time-honest ("as-of") evaluation of Medicare fraud flags with delayed exclusion labels

Status: FROZEN DRAFT for commit, written 7 Oct 2026 (Cowork) from `Paper2_NewMethods_Memo_Oct5.md` §4. It becomes the frozen
protocol when Claude Code commits it to `experiments/cms_leie/asof/protocol_T_frozen.md` and records the commit hash and the
file's SHA-256 in `asof/PROTOCOL_HASH.txt`. No model under this protocol is trained before that commit. Deviations after the
commit are appended with a date and a reason, never edited in place. The protocol is also the text of an optional OSF
preregistration (before 16 Nov 2026).

## 0. Question and claims
The CMS–LEIE literature line (and the Paper 2 arm, HCMS-D-26-00820) evaluates fraud flags with hindsight labels: a provider
excluded in 2024 is a positive in every earlier year, and cross-validation lets a model learn from exclusions that had not
happened when the flag would have been raised. Protocol T measures what a program-integrity unit faces: a model trained only
on exclusions known at the start of year t, scored on year t, judged by exclusions that come later. Claims are limited to the
pre-specified decision rules in §7. Exclusion is the label; the model refers; people decide. No provider is named anywhere.

## 1. Data (identical to the arm; nothing re-downloaded)
- Rows: the arm's kept US provider-years, CY2013–2024, 13,527,801 rows (`data/labels_<year>.parquet`, `drop_post_excl = 0`),
  from the CMS *Medicare Physician & Other Practitioners — by Provider* files and the LEIE of 10 Sep 2026 (SHA-256 in
  `out/download_manifest.json`). No LEIE file dated after 10 Sep 2026 is used (prospective-test rule).
- Label source: eventual `is_fraud` (authorities 1128(a)(1), (a)(2), (a)(3), (b)(4), (b)(7)) as in the arm. New column
  `excl_end_date` per NPI = the earliest `end_dt` (= min(EXCLDATE, WAIVERDATE) of the record, as `build_labels.py` computes it)
  among the NPI's fraud-authority LEIE records; `excl_year` as in the arm (year of `end_dt`, +1 if month ≥ 7).
- Features: the arm's primary set, 55 columns in `out/feature_list.json` order, from `data/feat_<year>.parquet`; the arm's
  category vocabularies (`out/category_vocab.json`). Calendar year is never a feature.

## 2. As-of years and sets
- t ∈ {2018, 2019, 2020, 2021, 2022} (five forward years; every t has ≥ 3 years of label follow-up to 10 Sep 2026).
- Training rows at t: all kept rows with year ≤ t − 1.
- Known-positive label at t: `known_t = 1` iff `is_fraud = 1` and `excl_end_date < 1 January t`. Every other training row is
  a negative at t, including rows of providers excluded on or after 1 January t (the hidden positives — this is the point).
- Test rows at t: every kept row with year = t (natural prevalence). Primary test label: eventual `is_fraud`. Secondary test
  label: `is_fraud_3yr_t = 1` iff `is_fraud = 1` and `excl_year ≤ t + 3`.

## 3. Sampling and resampling (as the arm, seeds shifted by t)
- Training set at t: all known positives + 20 negatives per known positive, sampled without replacement from the training
  negatives with `numpy.random.default_rng(42 + t)`; then `SMOTEENN(random_state = 42)`; the resampled matrix is cached once
  per (t, variant) so both models train on identical rows.
- Scaling (variant A): RobustScaler convention of the arm — (x − median)/IQR, IQR 0 → 1 — **with the statistics computed on the
  training rows (years ≤ t − 1) only** and applied to year t. (The arm computed them over all years; Protocol T does not.)

## 4. Models (hyper-parameters identical to the arm, `random_state = 42`)
- Gradient Boosting: 500 trees, max_depth 6, learning_rate 0.05, subsample 0.8 (warm-start increments of 30, pickled between
  increments, as `train_model.py`). Random Forest: 500 trees, max_depth 20, min_samples_leaf 5, class_weight balanced.
- Two feature variants: **A** = arm features, scaler of §3. **B** = peer-normalised features: for each of the 48 numeric columns
  (37 published numerics + 11 derived ratios), (x − median_{year, provider type})/IQR_{year, provider type} with IQR 0 → 1,
  computed on the rows of the same calendar year and provider type only (no labels used; available at scoring time); the two
  indicators and the five categoricals unchanged. Provider types with fewer than 30 rows in a year fall back to the year's
  all-type median/IQR.
- Fits per t: GB-A, RF-A, GB-B, RF-B (20 fits in all). Scores for all year-t rows are saved (`results/asof/scores_<t>.parquet`:
  npi, year, is_fraud, is_fraud_3yr, p_gb_A, p_rf_A, p_gb_B, p_rf_B); the file is kept outside the repository and its SHA-256
  recorded.

## 5. Metrics per t (from the saved scores; `asof_metrics.py`)
- Ranking within year t: score descending, NPI ascending (ties), as `capacity_table.py`. Budgets k ∈ {0.5 %, 1 %, 2 %, 5 %} of
  year-t rows, floor(k · n_t).
- recall@k, precision@k, lift@k = precision@k / prevalence_t; dollar-weighted recall@k = Σ Tot_Mdcr_Pymt_Amt over fraud rows in
  the top k ÷ Σ over all fraud rows of year t; AUC-PR; AUC-ROC. Each for the primary and the 3-year label.
- Hindsight comparison: the same metrics for year t computed from the arm's out-of-fold probabilities (`results/oof_fold*.parquet`,
  GB and RF, primary run) restricted to year t. Hindsight-optimism ratio_t = hindsight lift@1 % ÷ as-of lift@1 % (GB-A primary;
  RF-A reported).
- Label-completeness context: share of the eventual positives among the training rows (years ≤ t − 1) that were known at 1 January t (the as-of completeness table of the memo, recomputed from `data/asof_known.parquet`).

## 6. Triage and allocation comparisons (as-of GB-A scores; no retraining)
- Ranking rules at k = 1 %: (i) p; (ii) p × Tot_Mdcr_Pymt_Amt (expected payments at risk); (iii) Tot_Mdcr_Pymt_Amt alone.
  Outputs: fraud rows caught, distinct later-excluded NPIs, dollar-weighted recall.
- Yield-aware specialty allocation at k = 1 %: for year t, yield_s = (known_t positives among the top-1 % rows of year t − 1 of
  provider type s, ranked by the model built at t) ÷ (top-1 % rows of type s in year t − 1); types with fewer than 50 flagged
  rows in t − 1 take the pooled yield. Budget for type s in year t = floor(k · n_t · n_{s,t} · yield_s ÷ Σ_s' n_{s',t} · yield_s');
  unused budget (types with fewer eligible rows than budget) is returned to the global ranking; if every yield estimate is 0 the
  allocation reduces to the global ranking. Compared with the global top-1 % ranking on recall@1 % and dollar-weighted
  recall@1 %. The hindsight-optimism ratio is undefined (reported as null) in a year whose as-of lift@1 % is 0; T1 compares
  the two lifts directly.

## 7. Decision rules (reported as met / not met; never reinterpreted)
- T1 "hindsight optimism exists": as-of lift@1 % (GB-A) < hindsight lift@1 % in ≥ 4 of 5 years.
- T2 "peer normalisation helps": GB-B beats GB-A on recall@1 % in ≥ 4 of 5 years AND the five-year mean difference's bootstrap
  95 % interval excludes 0.
- T3 "dollar triage helps": p × payment beats p on dollar-weighted recall@1 % in 5 of 5 years.
- T4 "yield-aware allocation helps": beats the global ranking on recall@1 % in ≥ 4 of 5 years AND the five-year mean
  difference's bootstrap 95 % interval excludes 0.
- Bootstrap: providers are the resampling unit, resampled with replacement within each year t independently
  (`default_rng(2026 + t)`), B = 1,000; paired across the compared rules/variants; the statistic is the mean over the five
  years of the per-year difference; percentile 2.5/97.5 interval. No significance test is run on five numbers alone.

## 8. Replication and reporting
- Primary run on the Mac (Claude Code); replication in the cloud; the data pipeline is byte-identical (SHA-256 of every
  input and output recorded); model fits agree within platform floating point as documented for the arm
  (`results/mac_vs_cloud_v32_diffs.md`).
- Every number in the paper traces to a saved run output (Rule 0d); the protocol's SHA-256 and commit hash are printed in
  the paper; the prior-art log (`PriorArt_Log_NewMethods.md`) is appended before any novelty word is used.
- The prospective CY2024 test (OSF registration submitted 6 Oct 2026; commitment `commitment_cy2024_full_gb.json`,
  SHA-256 lists) is reported alongside, with its own pre-specified rule (exact one-sided hypergeometric p < 0.01 at the
  Jan/Apr/Jul 2027 LEIE files).

## 9. What is not in Protocol T
- No new architectures; no HCPCS-mix features (candidate M6 waits for Paper S's outcome-file download and would be a dated
  amendment); no claim about recoveries, prevented losses or any real program; no provider named.

## 10. Order of execution
`asof_labels.py` (adds `excl_end_date`, writes `data/asof_known.parquet` with known_t flags for all t) → `asof_build.py t variant`
(scaler statistics on ≤ t − 1; sampling; SMOTE-ENN; cache) → `asof_train.py t variant model` (resumable warm-start; scores
year t) → `asof_metrics.py` (all of §5–§7; `results/asof/summary.json` and `summary.md`) → gate check against this file
before anything is reported.

## Deviations (appended after the freeze commit 8cadcb87f602fc92e877976262351d31b16967e1 of 8 Oct 2026 05:18 UTC; the frozen text is the file content up to, not including, the blank line before this heading — sha256 4bdbad3d02d1e41331b062a93d645d441345f6693ea8237e101b03f8f804c061, as in PROTOCOL_HASH.txt)
- 2026-10-08 05:20 UTC — §8 "Primary run on the Mac (Claude Code); replication in the cloud": the primary run is executed in the Cowork cloud workspace (the arm's data and out-of-fold results are byte-identical there, `results/mac_vs_cloud_v32_diffs.md`), because Bharath delegated execution to the Cowork session on 8 Oct 2026; the Mac run becomes the replication. Reason: availability of compute without Bharath's attendance. No other section changes.
