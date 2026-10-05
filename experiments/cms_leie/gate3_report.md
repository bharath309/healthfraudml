## 1. Folds
- Providers in more than one fold: **0**
- Rows per fold: {1: 2705561, 2: 2705560, 3: 2705560, 4: 2705560, 5: 2705560}

## 2–3. Fold metrics from stored probabilities, and summary means/SDs from fold files
| run | model | max |Δ| fold JSON vs recomputed | max |Δ| summary vs fold files |
|---|---|---|---|
| primary_is_fraud | gb | 0.0e+00 | 0.0e+00 |
| primary_is_fraud | rf | 0.0e+00 | 0.0e+00 |
| primary_is_fraud | mlp | 0.0e+00 | 0.0e+00 |
| primary_is_fraud | nb | 0.0e+00 | 0.0e+00 |
| primary_is_fraud | svm | 0.0e+00 | 0.0e+00 |
| primary_is_fraud | ens | 0.0e+00 | 0.0e+00 |
| primary_is_fraud_mandatory | gb | 0.0e+00 | 0.0e+00 |
| primary_is_fraud_mandatory | rf | 0.0e+00 | 0.0e+00 |
| primary_is_fraud_5yr | gb | 0.0e+00 | 0.0e+00 |
| primary_is_fraud_5yr | rf | 0.0e+00 | 0.0e+00 |
| sensitivity_cc_2017plus_is_fraud | gb | 0.0e+00 | 0.0e+00 |
| sensitivity_cc_2017plus_is_fraud | rf | 0.0e+00 | 0.0e+00 |
| primary_plus_year_is_fraud | gb | 0.0e+00 | 0.0e+00 |
| primary_plus_year_is_fraud | rf | 0.0e+00 | 0.0e+00 |
| primary_is_fraud_r50 | gb | 0.0e+00 | 0.0e+00 |
| primary_is_fraud_r50 | rf | 0.0e+00 | 0.0e+00 |

## 4. Gradient Boosting threshold study (all 99 grid thresholds)
| run | max |precision − πR/(πR+(1−π)FPR)| | grid optimum (mine / summary) | nested F1 (mine / summary) |
|---|---|---|---|
| primary_is_fraud | 6.9e-18 | 0.89 / 0.89 | 0.0445 / 0.0445 |
| primary_is_fraud_mandatory | 1.4e-17 | 0.90 / 0.90 | 0.0505 / 0.0505 |
| primary_is_fraud_5yr | 1.7e-18 | 0.63 / 0.63 | 0.0138 / 0.0138 |
| sensitivity_cc_2017plus_is_fraud | 1.4e-17 | 0.79 / 0.79 | 0.0298 / 0.0298 |
| primary_plus_year_is_fraud | 2.8e-17 | 0.86 / 0.86 | 0.0512 / 0.0512 |
| primary_is_fraud_r50 | 1.4e-17 | 0.80 / 0.80 | 0.0388 / 0.0388 |

## 5. Rank orders vs the paper (primary run, six models)

**F1** — CMS–LEIE: Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > SVM (RBF) > Naive Bayes
- vs synthetic: Gradient Boosting > Neural Network (MLP) > Random Forest > AdaBoost-Voting Ensemble > SVM (RBF) > Naive Bayes — τ = 0.73 (p = 0.06), ρ = 0.83; top-2 identical: **no**; rule (top-2 identical and τ ≥ 0.6): **not met**
- vs kaggle_medicare: Gradient Boosting > Neural Network (MLP) > Random Forest > AdaBoost-Voting Ensemble > SVM (RBF) > Naive Bayes — τ = 0.73 (p = 0.06), ρ = 0.83; top-2 identical: **no**; rule (top-2 identical and τ ≥ 0.6): **not met**

**AUC-PR** — CMS–LEIE: Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > Naive Bayes > SVM (RBF)
- vs synthetic: Gradient Boosting > Random Forest > AdaBoost-Voting Ensemble > Neural Network (MLP) > SVM (RBF) > Naive Bayes — τ = 0.87 (p = 0.02), ρ = 0.94; top-2 identical: **yes**; rule (top-2 identical and τ ≥ 0.6): **met**
- vs kaggle_medicare: Neural Network (MLP) > Random Forest > Gradient Boosting > AdaBoost-Voting Ensemble > SVM (RBF) > Naive Bayes — τ = 0.33 (p = 0.47), ρ = 0.54; top-2 identical: **no**; rule (top-2 identical and τ ≥ 0.6): **not met**

## 6. Sensitivities — effective class balance
| run | train rows | train positives | effective neg:pos | test prevalence |
|---|---|---|---|---|
| primary_is_fraud | 938,196 | 44,676 | 20.0:1 | 0.0826% |
| primary_is_fraud_mandatory | 938,196 | 25,444 | 35.9:1 | 0.0470% |
| primary_is_fraud_5yr | 938,196 | 19,424 | 47.3:1 | 0.0359% |
| sensitivity_cc_2017plus_is_fraud | 648,802 | 20,040 | 31.4:1 | 0.0527% |
| primary_plus_year_is_fraud | 938,196 | 44,676 | 20.0:1 | 0.0826% |
| primary_is_fraud_r50 | 2,278,476 | 44,676 | 50.0:1 | 0.0826% |

## 7. Chronic-condition fields: why `cc_missing` is almost always 1
| year | rows | any CC blank | all 25 blank | median blank fields | rows with Tot_Benes < 11 |
|---|---|---|---|---|---|
| 2017 | 1,088,687 | 93.3% | 0.0% | 6 | 0.0% |
| 2018 | 1,121,462 | 93.1% | 0.0% | 7 | 0.0% |
| 2019 | 1,155,870 | 93.1% | 0.0% | 7 | 0.0% |
| 2020 | 1,161,542 | 94.5% | 0.0% | 7 | 0.0% |
| 2021 | 1,198,754 | 94.6% | 0.0% | 7 | 0.0% |
| 2022 | 1,230,293 | 95.0% | 0.0% | 7 | 0.0% |
| 2023 | 1,259,343 | 95.2% | 0.0% | 7 | 0.0% |
| 2024 | 1,296,739 | 95.3% | 0.0% | 7 | 0.0% |

Share blank per field (mean over 2017–2024), most to least: BH_PTSD_V1 66%; BH_ADHD_OthCD_V1 65%; BH_PD_V1 65%; PH_Parkinson_V2 60%; BH_Schizo_OthPsy_V1 57%; BH_Bipolar_V1 54%; BH_Alcohol_Drug_V1 44%; BH_Alz_NonAlzdem_V2 41%; PH_Stroke_TIA_V2 39%; BH_Tobacco_V1 35%; PH_Asthma_V2 34%; PH_Osteoporosis_V2 30%; PH_Cancer6_V2 29%; PH_HF_NonIHD_V2 28%; PH_Afib_V2 26%; PH_COPD_V2 26%; PH_CKD_V2 22%; PH_IschemicHeart_V2 19%; BH_Depress_V1 18%; BH_Anxiety_V1 16%; PH_Diabetes_V2 16%; BH_Mood_V2 15%; PH_Arthritis_V2 9%; PH_Hyperlipidemia_V2 5%; PH_Hypertension_V2 4%
