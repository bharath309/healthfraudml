# Gate 3 pre-check — CMS–LEIE arm, Mac run (5 Oct 2026, Claude Code)

Every item of Gate 3 in `Paper2_ML_Comparison/cms_leie_arm/Verification_Checklist_CMS_LEIE_Arm_Oct3.md`, run with an
independent script (`gate3_check.py`; full output `gate3_report.md`, `gate3_report.json`). This is a pre-check for
Cowork's sign-off and Bharath's decision; it does not sign the checklist.

## Results

| Gate 3 item | Result |
|---|---|
| No NPI in two folds; fold sizes as reported | **Pass.** 0 providers in more than one fold; 2,705,560–2,705,561 test rows per fold. |
| One GB fold reproduced to four decimals, or the difference explained and within 0.003 F1 | **Pass (explained).** Mac vs Cowork GB F1 differs by at most 0.00058 across folds (RF 0.00008, MLP 0.00012, SVM 0.00001, ensemble 0.00007). Cause: cross-platform floating point, same library versions (`MAC_VS_COWORK.md`). Mac reruns are bit-identical. |
| `summary.json` means/SDs recomputed from fold files | **Pass.** Max difference 0.0 for every model in all six runs. Fold metrics recomputed from stored probabilities also match exactly (0.0). |
| Precision identity on every threshold row; nested selection reproduced | **Pass.** Identity holds to ≤ 2.8e-17 on all 99 grid thresholds in all six runs; grid optimum and nested F1 reproduced exactly. |
| Rank orders vs Tables 1–2; Kendall/Spearman saved | **Done — see decision rule below.** (The Mac `summarise.py` could not find the paper files at Cowork's `/mnt/user-data/...` path, so its comparison was empty; `gate3_check.py` uses the repo copies `results/{synthetic,kaggle_medicare}/experiment_results.json`, which give the same paper orders Cowork printed.) |
| `results_report.md` numbers located in files | No `results_report.md` was written on the Mac; every number in `MAC_VS_COWORK.md` and here is generated from `summary.json`/fold files by script. |
| Sensitivities present, same checks | **Pass.** All five present, all checks above pass on each. |
| Limitations list | Drafted below. |

## Decision rule (pre-specified in the checklist): "top-2 identical and Kendall τ ≥ 0.6" on F1 and AUC-PR

| Metric | CMS–LEIE order | vs synthetic arm | vs Kaggle arm |
|---|---|---|---|
| F1 | GB > RF > Ens > MLP > SVM > NB | paper GB > **MLP** > RF …; τ = 0.73 (p = 0.06) — top-2 differ → **not met** | same paper order; τ = 0.73 — **not met** |
| AUC-PR | GB > RF > Ens > MLP > NB > SVM | paper GB > RF > Ens > MLP > SVM > NB; τ = 0.87 (p = 0.02) — **met** | paper MLP > RF > GB …; τ = 0.33 (p = 0.47) — **not met** |

**Reading: the rule is met in 1 of 4 comparisons.** GB is first everywhere except Kaggle AUC-PR, but RF, not MLP, is
second on real labels. By the checklist's own text, "Ranking does not hold … approve v31 unchanged; the arm becomes the
revision response or the next paper." Decision is Bharath's, on Cowork's reading.

## Effective class balance in the sensitivities (protocol note 13:20 UTC: report it)

| Run | Train positives | Effective neg:pos | Test prevalence |
|---|---|---|---|
| Primary | 44,676 | 20.0:1 | 0.0826% |
| Mandatory exclusions | 25,444 | 35.9:1 | 0.0470% |
| 5-year window | 19,424 | 47.3:1 | 0.0359% |
| Chronic conditions 2017–2024 | 20,040 | 31.4:1 | 0.0527% |
| Year as feature | 44,676 | 20.0:1 | 0.0826% |
| 50:1 negatives | 44,676 | 50.0:1 | 0.0826% |

## `cc_missing`: blanks are hidden small counts, not unknown values

- 93–95% of provider-years (2017–2024) have at least one blank chronic-condition field; none has all 25 blank; median
  7 blank fields. Rare conditions are blank most often (PTSD 66%, ADHD 65%, personality disorders 65%, Parkinson 60%);
  hypertension 4%, hyperlipidemia 5%.
- Tested on CY2024: a published value is either **0** or a count of **≥ 11 patients**; nonzero counts under 11
  essentially never appear (≤ 0.2%, consistent with rounding). The CMS data dictionary states counts under 11 are
  suppressed and **percentages from 75% to 100% are top-coded at 75%**.
- So a blank means "1 to 10 patients", a small positive share. `build_features.py` fills blanks with 0 (biased low) and
  `cc_missing` ("any blank") is close to constant, so it carries almost no information.
- The same applies to the primary feature set: demographic counts are blank-filled with 0 (`build_features.py` line 52);
  Cowork already replaced the near-constant "any demographic missing" flag with a count (line 26).
- Better encoding for any rerun: per-field midpoint imputation (5.5 / Tot_Benes) plus per-field blank indicators, with
  bounds 1 and 10 as a sensitivity; treat 75 as "≥ 75".

## Limitations list for the paper (from the data, Rule 0b)
1. LEIE is a 10 Sep 2026 snapshot; exclusions are an incomplete proxy for fraud (licence-based authorities included in the primary label).
2. Exclusion lag: recent years carry fewer positives.
3. CMS suppression: rows from ≤ 10 beneficiaries are absent; chronic-condition and demographic cells covering 1–10
   patients are blank and were filled with 0; chronic-condition percentages are top-coded at 75%.
4. Provider-year grain; no claim-level information.
5. Training negatives pre-sampled at 20:1 (50:1 sensitivity); effective ratios 31–47:1 under the alternative labels.
6. SVC fitted on a 30,000-row subsample.
7. Provider-disjoint folds; results do not test temporal generalisation (year as a feature raises GB F1 from 0.0175 to 0.0245).
8. Cross-platform reproducibility to about three decimals; exact within one platform.
