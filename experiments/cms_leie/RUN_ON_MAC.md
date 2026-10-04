# Re-running the CMS–LEIE arm on the Mac (Claude Code) — the verification run and the completion of SVM / ensemble / sensitivities

Written 3 Oct 2026 after the Cowork cloud run produced the primary results for Gradient Boosting, Random Forest, MLP and
Naive Bayes on all five folds (`results/primary_is_fraud/summary.md`). Everything is deterministic: same files (check the
SHA-256 values in `out/download_manifest.json`), same seeds, same library versions (`out/environment.txt`:
scikit-learn 1.8.0, imbalanced-learn 0.14.2, numpy 2.4.4, pandas 3.0.2, pyarrow 25.0.1). Expect four-decimal agreement.

Paste into Claude Code, working in `EB1A prep/GitHub_Framework` on branch `cms-leie-arm`:

1. Copy the folder `Paper2_ML_Comparison/cms_leie_arm/run/` to `experiments/cms_leie/` in the repository (scripts, `out/`,
   `protocol_frozen.md`, this file). Add `experiments/cms_leie/data/` and `experiments/cms_leie/state/` to `.gitignore`.
   Create a virtual environment with exactly the versions in `out/environment.txt` (`pip install scikit-learn==1.8.0
   imbalanced-learn==0.14.2 numpy==2.4.4 pandas==3.0.2 pyarrow==25.0.1 scipy==1.17.1`).
2. `python3 download.py leie 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024` — about 5.6 GB of CSV, converted
   to parquet (1.4 GB) and deleted. **Compare every SHA-256 with `out/download_manifest.json`.** The LEIE is replaced
   monthly: if the OIG file has rolled over since 10 Sep 2026, its hash will differ; in that case stop and report — the
   labels must come from the same snapshot (Cowork keeps the 10 Sep 2026 file: 15,608,468 bytes,
   SHA-256 `175352046e438bc477fe0f8985c645c3eb492114b449e13a5700b30e00d27417`; it can be copied over from the cloud run).
3. `python3 build_labels.py` → `out/label_report.json` must match the Cowork copy exactly (11,169 fraud provider-years;
   per-year counts; 29 dropped rows). `python3 build_features.py` → `out/scaler_stats.json` identical.
   `python3 make_folds.py` → `out/folds_report.json` identical (fold sizes and positives).
4. Primary run, in parallel across cores (each command is independent once its fold is resampled):
   `for k in 1 2 3 4 5; do python3 resample_fold.py $k; done`
   `for m in gb rf mlp nb rf200 gb200 svm svc200; do for k in 1 2 3 4 5; do python3 train_model.py $m $k --budget 1000000; done; done`
   (run the (m, k) jobs in parallel with `xargs -P <cores>` or `parallel`; each job is single-threaded except RF). SVM and
   svc200 prediction over 2.7 M rows takes ~45 min per fold per core on the cloud machine; splitting batches across cores
   is fine because the state file is per (model, fold).
5. `python3 summarise.py` → `results/primary_is_fraud/summary.{json,md}`. Compare with the Cowork copy: GB, RF, MLP, NB
   must agree to four decimals per fold; SVM and the ensemble are new.
6. Sensitivities (protocol_frozen.md §Sensitivities), GB and RF only unless stated:
   - `--label is_fraud_mandatory` (all six models): `resample_fold.py k primary is_fraud_mandatory`, then
     `train_model.py <m> k --label is_fraud_mandatory`, then `summarise.py --label is_fraud_mandatory`.
   - `--label is_fraud_5yr` likewise.
   - `--set sensitivity_cc_2017plus` (rows 2017–2024): `resample_fold.py k sensitivity_cc_2017plus`, then
     `train_model.py <m> k --set sensitivity_cc_2017plus`, then `summarise.py --set sensitivity_cc_2017plus`.
   - Year as a feature and the 50:1 ratio need two one-line changes (add `year` to the feature list in `cmsarm.py`; set
     RATIO=50 in `make_folds.py` writing to a separate folder) — make them on a copy and record them in the results folder.
7. Commit `out/`, `results/**/*.json`, `results/**/summary.md`, `protocol_frozen.md`, the scripts and this file. Never commit
   `data/`, `state/` or the `.npy` probability arrays (they are 10 MB each; keep them on disk and in the Mac folder).
8. Report back the `summary.md` files and the SHA-256 comparison. Cowork performs the gate-3 checks before anything
   enters the manuscript.

## Appended 3 Oct 2026, 18:40 UTC — scripts now cover every sensitivity without edits
- `run_sensitivity.sh <set> <label> [ratio]` runs SMOTE-ENN + GB + RF for five folds and summarises. The five
  pre-specified sensitivities are: `./run_sensitivity.sh primary is_fraud_mandatory`; `./run_sensitivity.sh primary
  is_fraud_5yr`; `./run_sensitivity.sh sensitivity_cc_2017plus is_fraud` (rows 2017–2024; `cc_missing` scaler statistic
  added to `out/scaler_stats.json`); `./run_sensitivity.sh primary_plus_year is_fraud` (feature set = primary + `year`,
  `year` statistic added); `./run_sensitivity.sh primary is_fraud 50` after `python3 make_folds.py 50` (reuses the saved
  fold assignment, writes `train_<k>_r50.parquet` and `out/folds_report_r50.json`: 446,750–446,800 negatives per fold).
- `train_model.py` takes `--ratio`; `resample_fold.py` takes a fourth argument (ratio); `summarise.py` takes `--ratio`;
  results go to `results/<set>_<label>[_r<ratio>]/`.
- SVC scoring uses `fastsvc.py` (exact; see protocol_frozen.md 17:35 UTC note). The ensemble's SVC member equals the
  standalone SVM; `train_model.py svc200 k` recomputes it if you want the independent check, or copy the SVM arrays.
- Cowork's cloud run (3 Oct): primary results complete for all six architectures; sensitivities running in the order
  mandatory → chronic-condition set → 5-year window → year feature → 50:1, resumable; whatever is unfinished when you run
  this is simply re-run here from the same files.
- The Cowork desktop workspace on this Mac (the Linux VM behind Cowork's file tools) is not a usable verification
  environment: Python 3.10 (scikit-learn 1.8.0 not installable), 3 GB RAM, processes killed after each 3-minute call.
  Claude Code in a normal shell is.
