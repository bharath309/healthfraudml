#!/usr/bin/env python3
"""
Comparative Analysis of ML Architectures for Healthcare Claims Fraud Detection.

Supports two data modes:
  1. Synthetic — generates a 50K-claim dataset with controllable fraud patterns
  2. Kaggle   — loads the Healthcare Provider Fraud Detection dataset
                (https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

Trains 8 ML models, evaluates using F1, AUC-PR, MCC, and generates SHAP explanations.

Usage:
  python run_comparison.py                    # synthetic data only
  python run_comparison.py --kaggle           # Kaggle data only
  python run_comparison.py --both             # both datasets side-by-side

Paper: "Comparative Analysis of Machine Learning Architectures for Healthcare
Claims Fraud Detection with Explainable Outputs"

Author: Bharath Kumar Bahudhoddi, Ph.D.
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")


# ============================================================
# 1. SYNTHETIC DATASET GENERATOR
# ============================================================

def generate_synthetic_claims(n_samples=50000, fraud_rate=0.032, random_state=42):
    """
    Generate synthetic healthcare claims dataset reflecting real-world
    statistical properties from CMS Medicare Provider Utilization data.

    Parameters
    ----------
    n_samples : int
        Total number of claims to generate.
    fraud_rate : float
        Proportion of fraudulent claims (default 3.2%).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Claims dataset with features and fraud labels.
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # --- CPT code universe with severity and price benchmarks ---
    cpt_codes = {
        "99281": {"desc": "ED Visit L1", "severity": 1, "fair_min": 150, "fair_max": 400, "category": "EM"},
        "99282": {"desc": "ED Visit L2", "severity": 2, "fair_min": 300, "fair_max": 700, "category": "EM"},
        "99283": {"desc": "ED Visit L3", "severity": 3, "fair_min": 500, "fair_max": 1200, "category": "EM"},
        "99284": {"desc": "ED Visit L4", "severity": 4, "fair_min": 1000, "fair_max": 2200, "category": "EM"},
        "99285": {"desc": "ED Visit L5", "severity": 5, "fair_min": 1500, "fair_max": 3500, "category": "EM"},
        "99213": {"desc": "Office Visit L3", "severity": 3, "fair_min": 150, "fair_max": 350, "category": "EM"},
        "99214": {"desc": "Office Visit L4", "severity": 3, "fair_min": 200, "fair_max": 450, "category": "EM"},
        "99215": {"desc": "Office Visit L5", "severity": 4, "fair_min": 300, "fair_max": 600, "category": "EM"},
        "12001": {"desc": "Wound Repair Simple", "severity": 1, "fair_min": 300, "fair_max": 800, "category": "SURG"},
        "12002": {"desc": "Wound Repair 2.6-7.5cm", "severity": 2, "fair_min": 400, "fair_max": 1000, "category": "SURG"},
        "56420": {"desc": "Bartholin Gland I&D", "severity": 2, "fair_min": 400, "fair_max": 1200, "category": "SURG"},
        "29881": {"desc": "Knee Arthroscopy", "severity": 3, "fair_min": 3000, "fair_max": 8000, "category": "SURG"},
        "43239": {"desc": "Upper GI Endoscopy", "severity": 3, "fair_min": 1500, "fair_max": 4000, "category": "SURG"},
        "70553": {"desc": "Brain MRI w/wo contrast", "severity": 3, "fair_min": 1000, "fair_max": 3500, "category": "DIAG"},
        "71046": {"desc": "Chest X-ray 2 views", "severity": 1, "fair_min": 100, "fair_max": 400, "category": "DIAG"},
        "80053": {"desc": "Comprehensive Metabolic Panel", "severity": 1, "fair_min": 30, "fair_max": 150, "category": "LAB"},
        "85025": {"desc": "Complete Blood Count", "severity": 1, "fair_min": 20, "fair_max": 100, "category": "LAB"},
        "90837": {"desc": "Psychotherapy 60 min", "severity": 2, "fair_min": 150, "fair_max": 400, "category": "BEHAV"},
        "96372": {"desc": "Therapeutic Injection", "severity": 1, "fair_min": 50, "fair_max": 200, "category": "ADMIN"},
        "97110": {"desc": "Therapeutic Exercises", "severity": 1, "fair_min": 50, "fair_max": 200, "category": "PT"},
    }

    cpt_list = list(cpt_codes.keys())
    cpt_weights = [0.05, 0.08, 0.15, 0.10, 0.05, 0.12, 0.10, 0.03,
                   0.03, 0.02, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05,
                   0.04, 0.02, 0.02, 0.02]

    # Provider universe
    n_providers = 500
    provider_ids = [f"P{i:04d}" for i in range(n_providers)]
    specialties = ["Emergency Medicine", "Internal Medicine", "Family Practice",
                   "Orthopedics", "Gastroenterology", "Radiology", "Pathology",
                   "Psychiatry", "Physical Therapy", "General Surgery"]
    provider_specialty = {pid: np.random.choice(specialties) for pid in provider_ids}

    # Patient universe
    n_patients = 15000
    patient_ids = [f"PAT{i:06d}" for i in range(n_patients)]
    patient_ages = {pid: int(np.random.normal(55, 18)) for pid in patient_ids}
    patient_ages = {k: max(1, min(99, v)) for k, v in patient_ages.items()}
    patient_gender = {pid: np.random.choice(["M", "F"]) for pid in patient_ids}
    regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
    patient_region = {pid: np.random.choice(regions) for pid in patient_ids}

    records = []

    # --- Generate legitimate claims ---
    for _ in range(n_legit):
        cpt = np.random.choice(cpt_list, p=cpt_weights)
        ref = cpt_codes[cpt]
        amount = np.random.uniform(ref["fair_min"], ref["fair_max"])
        if np.random.random() < 0.05:
            amount *= np.random.uniform(1.0, 1.15)

        provider = np.random.choice(provider_ids)
        patient = np.random.choice(patient_ids)

        records.append({
            "cpt_code": cpt,
            "cpt_category": ref["category"],
            "cpt_severity": ref["severity"],
            "amount": round(amount, 2),
            "fair_max": ref["fair_max"],
            "provider_id": provider,
            "specialty": provider_specialty[provider],
            "patient_id": patient,
            "patient_age": patient_ages[patient],
            "patient_gender": patient_gender[patient],
            "region": patient_region[patient],
            "num_procedures": np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]),
            "is_fraud": 0,
            "fraud_type": "none",
        })

    # --- Generate fraudulent claims ---
    fraud_types = ["upcoding", "unbundling", "phantom", "duplicate", "anomalous"]
    fraud_weights = [0.35, 0.25, 0.20, 0.12, 0.08]

    for _ in range(n_fraud):
        fraud_type = np.random.choice(fraud_types, p=fraud_weights)
        provider = np.random.choice(provider_ids[:50])
        patient = np.random.choice(patient_ids)

        if fraud_type == "upcoding":
            cpt = np.random.choice(["99285", "99284", "99215"])
            ref = cpt_codes[cpt]
            amount = np.random.uniform(ref["fair_max"] * 1.2, ref["fair_max"] * 2.5)
            severity = ref["severity"]
            num_proc = np.random.choice([2, 3])
        elif fraud_type == "unbundling":
            cpt = np.random.choice(["85025", "80053", "71046"])
            ref = cpt_codes[cpt]
            amount = np.random.uniform(ref["fair_max"] * 1.5, ref["fair_max"] * 4.0)
            severity = ref["severity"]
            num_proc = np.random.choice([4, 5, 6, 7])
        elif fraud_type == "phantom":
            cpt = np.random.choice(cpt_list, p=cpt_weights)
            ref = cpt_codes[cpt]
            amount = np.random.uniform(ref["fair_min"], ref["fair_max"])
            severity = ref["severity"]
            num_proc = np.random.choice([1, 2])
        elif fraud_type == "duplicate":
            cpt = np.random.choice(cpt_list, p=cpt_weights)
            ref = cpt_codes[cpt]
            amount = np.random.uniform(ref["fair_min"], ref["fair_max"])
            severity = ref["severity"]
            num_proc = 1
        else:  # anomalous
            cpt = np.random.choice(["29881", "43239", "70553"])
            ref = cpt_codes[cpt]
            amount = np.random.uniform(ref["fair_max"] * 0.8, ref["fair_max"] * 1.8)
            severity = ref["severity"]
            num_proc = np.random.choice([1, 2, 3])

        records.append({
            "cpt_code": cpt,
            "cpt_category": ref["category"],
            "cpt_severity": severity,
            "amount": round(amount, 2),
            "fair_max": ref["fair_max"],
            "provider_id": provider,
            "specialty": provider_specialty[provider],
            "patient_id": patient,
            "patient_age": patient_ages[patient],
            "patient_gender": patient_gender[patient],
            "region": patient_region[patient],
            "num_procedures": num_proc,
            "is_fraud": 1,
            "fraud_type": fraud_type,
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # --- Engineer provider-level aggregate features ---
    provider_stats = df.groupby("provider_id").agg(
        provider_claim_count=("amount", "count"),
        provider_mean_amount=("amount", "mean"),
        provider_std_amount=("amount", "std"),
        provider_fraud_hist=("is_fraud", "mean"),
    ).reset_index()
    provider_stats["provider_std_amount"] = provider_stats["provider_std_amount"].fillna(0)

    spec_stats = df.groupby("specialty").agg(
        specialty_mean_amount=("amount", "mean"),
        specialty_std_amount=("amount", "std"),
    ).reset_index()

    df = df.merge(provider_stats, on="provider_id", how="left")
    df = df.merge(spec_stats, on="specialty", how="left")

    df["specialty_deviation"] = (
        (df["provider_mean_amount"] - df["specialty_mean_amount"])
        / df["specialty_std_amount"].clip(lower=1)
    )
    df["amount_fair_ratio"] = df["amount"] / df["fair_max"].clip(lower=1)

    patient_visits = df.groupby("patient_id").size().reset_index(name="patient_visit_count")
    df = df.merge(patient_visits, on="patient_id", how="left")

    df["severity_amount_ratio"] = df["amount"] / (df["cpt_severity"] * 500 + 1)

    print(f"Generated {len(df)} claims: {df['is_fraud'].sum()} fraud "
          f"({df['is_fraud'].mean()*100:.1f}%)")
    print(f"Fraud types: {df[df['is_fraud']==1]['fraud_type'].value_counts().to_dict()}")

    return df


# ============================================================
# 2. KAGGLE DATASET LOADER
# ============================================================

def load_kaggle_dataset(data_dir):
    """
    Load the Kaggle Healthcare Provider Fraud Detection dataset.

    Download from: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
    Place these files in the data_dir:
      - Train_Beneficiarydata.csv
      - Train_Inpatientdata.csv
      - Train_Outpatientdata.csv
      - Train.csv  (provider-level fraud labels)

    Returns a provider-level DataFrame with engineered features and fraud labels.
    """
    required_files = [
        "Train_Beneficiarydata.csv",
        "Train_Inpatientdata.csv",
        "Train_Outpatientdata.csv",
        "Train.csv",
    ]
    for f in required_files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {f} in {data_dir}. Download the dataset from:\n"
                "https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis\n"
                f"and place all CSV files in: {data_dir}"
            )

    print("Loading Kaggle dataset files...")
    beneficiary = pd.read_csv(os.path.join(data_dir, "Train_Beneficiarydata.csv"))
    inpatient = pd.read_csv(os.path.join(data_dir, "Train_Inpatientdata.csv"))
    outpatient = pd.read_csv(os.path.join(data_dir, "Train_Outpatientdata.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "Train.csv"))

    # --- Merge inpatient + outpatient ---
    inpatient["ClaimType"] = "Inpatient"
    outpatient["ClaimType"] = "Outpatient"

    # Align columns — outpatient won't have some inpatient-only columns
    common_cols = list(set(inpatient.columns) & set(outpatient.columns))
    claims = pd.concat([inpatient[common_cols], outpatient[common_cols]],
                       ignore_index=True)

    # --- Merge with beneficiary data ---
    claims = claims.merge(beneficiary, on="BeneID", how="left")

    # --- Parse dates ---
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt",
                "DOB", "DOD"]:
        if col in claims.columns:
            claims[col] = pd.to_datetime(claims[col], errors="coerce")

    # --- Engineer features at the PROVIDER level ---
    # (The Kaggle labels are per-provider, not per-claim)

    # Claim-level features first
    if "ClaimStartDt" in claims.columns and "ClaimEndDt" in claims.columns:
        claims["ClaimDuration"] = (
            claims["ClaimEndDt"] - claims["ClaimStartDt"]
        ).dt.days.fillna(0)
    else:
        claims["ClaimDuration"] = 0

    claims["InscClaimAmtReimbursed"] = pd.to_numeric(
        claims.get("InscClaimAmtReimbursed", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    claims["DeductibleAmtPaid"] = pd.to_numeric(
        claims.get("DeductibleAmtPaid", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    claims["IPAnnualReimbursementAmt"] = pd.to_numeric(
        claims.get("IPAnnualReimbursementAmt", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    claims["OPAnnualReimbursementAmt"] = pd.to_numeric(
        claims.get("OPAnnualReimbursementAmt", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)

    # Count diagnosis and procedure codes per claim
    diag_cols = [c for c in claims.columns if c.startswith("ClmDiagnosisCode")]
    proc_cols = [c for c in claims.columns if c.startswith("ClmProcedureCode")]
    claims["NumDiagnoses"] = claims[diag_cols].notna().sum(axis=1) if diag_cols else 0
    claims["NumProcedures"] = claims[proc_cols].notna().sum(axis=1) if proc_cols else 0

    # Chronic condition columns
    chronic_cols = [c for c in claims.columns
                    if c.startswith("ChronicCond_") or c.startswith("RenalDiseaseIndicator")]
    if chronic_cols:
        for col in chronic_cols:
            claims[col] = pd.to_numeric(claims[col], errors="coerce").fillna(0)
        claims["NumChronicConditions"] = claims[chronic_cols].apply(
            lambda row: (row == 1).sum(), axis=1)
    else:
        claims["NumChronicConditions"] = 0

    # Age at claim
    if "DOB" in claims.columns and "ClaimStartDt" in claims.columns:
        claims["Age"] = (
            (claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.25
        ).fillna(65).clip(0, 110)
    else:
        claims["Age"] = 65

    # Is deceased
    claims["IsDeceased"] = claims["DOD"].notna().astype(int) if "DOD" in claims.columns else 0

    # --- Aggregate to provider level ---
    provider_agg = claims.groupby("Provider").agg(
        total_claims=("ClmAdmitDiagnosisCode", "count") if "ClmAdmitDiagnosisCode" in claims.columns
            else ("BeneID", "count"),
        unique_patients=("BeneID", "nunique"),
        mean_reimbursement=("InscClaimAmtReimbursed", "mean"),
        std_reimbursement=("InscClaimAmtReimbursed", "std"),
        total_reimbursement=("InscClaimAmtReimbursed", "sum"),
        mean_deductible=("DeductibleAmtPaid", "mean"),
        mean_claim_duration=("ClaimDuration", "mean"),
        std_claim_duration=("ClaimDuration", "std"),
        mean_num_diagnoses=("NumDiagnoses", "mean"),
        mean_num_procedures=("NumProcedures", "mean"),
        mean_chronic_conditions=("NumChronicConditions", "mean"),
        mean_age=("Age", "mean"),
        pct_deceased=("IsDeceased", "mean"),
    ).reset_index()

    provider_agg["std_reimbursement"] = provider_agg["std_reimbursement"].fillna(0)
    provider_agg["std_claim_duration"] = provider_agg["std_claim_duration"].fillna(0)
    provider_agg["claims_per_patient"] = (
        provider_agg["total_claims"] / provider_agg["unique_patients"].clip(lower=1)
    )

    # --- Merge with fraud labels ---
    labels["PotentialFraud"] = (labels["PotentialFraud"] == "Yes").astype(int)
    provider_agg = provider_agg.merge(labels, on="Provider", how="inner")

    print(f"Kaggle dataset loaded: {len(provider_agg)} providers, "
          f"{provider_agg['PotentialFraud'].sum()} fraudulent "
          f"({provider_agg['PotentialFraud'].mean()*100:.1f}%)")

    return provider_agg


def preprocess_kaggle(df):
    """Prepare Kaggle provider-level features for ML models."""
    from sklearn.preprocessing import RobustScaler

    feature_cols = [
        "total_claims", "unique_patients", "mean_reimbursement",
        "std_reimbursement", "total_reimbursement", "mean_deductible",
        "mean_claim_duration", "std_claim_duration", "mean_num_diagnoses",
        "mean_num_procedures", "mean_chronic_conditions", "mean_age",
        "pct_deceased", "claims_per_patient",
    ]

    X = df[feature_cols].values.astype(np.float64)
    y = df["PotentialFraud"].values

    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_cols, scaler


# ============================================================
# 3. PREPROCESSING & FEATURE ENGINEERING (Synthetic)
# ============================================================

def preprocess(df):
    """Prepare features and labels for ML models."""
    from sklearn.preprocessing import LabelEncoder, RobustScaler

    feature_cols = [
        "cpt_severity", "amount", "fair_max", "num_procedures",
        "patient_age", "provider_claim_count", "provider_mean_amount",
        "provider_std_amount", "specialty_deviation", "amount_fair_ratio",
        "patient_visit_count", "severity_amount_ratio",
    ]

    le_cpt = LabelEncoder()
    df["cpt_code_enc"] = le_cpt.fit_transform(df["cpt_code"])
    le_cat = LabelEncoder()
    df["cpt_category_enc"] = le_cat.fit_transform(df["cpt_category"])
    le_spec = LabelEncoder()
    df["specialty_enc"] = le_spec.fit_transform(df["specialty"])
    le_gender = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["patient_gender"])
    le_region = LabelEncoder()
    df["region_enc"] = le_region.fit_transform(df["region"])

    feature_cols += ["cpt_code_enc", "cpt_category_enc", "specialty_enc",
                     "gender_enc", "region_enc"]

    X = df[feature_cols].values.astype(np.float64)
    y = df["is_fraud"].values

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_cols, scaler


# ============================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================

def run_experiments(X, y, feature_names, n_folds=5, random_state=42):
    """
    Run 5-fold stratified CV for all models.
    Returns dict of {model_name: {metrics}} and feature importance info.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 matthews_corrcoef, average_precision_score)
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                  AdaBoostClassifier, VotingClassifier)
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    try:
        from imblearn.combine import SMOTEENN
        IMBLEARN_AVAILABLE = True
    except ImportError:
        IMBLEARN_AVAILABLE = False
        print("WARNING: imbalanced-learn not installed. Skipping SMOTE-ENN.")
        print("Install: pip install imbalanced-learn")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {}

    def get_models():
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_leaf=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1)
        svm = SVC(
            kernel="rbf", C=10, gamma="scale", class_weight="balanced",
            probability=True, random_state=random_state)
        ann = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            random_state=random_state)
        gb = GradientBoostingClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=random_state)
        nb = GaussianNB()
        voting = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=15,
                    class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=5,
                    learning_rate=0.05, random_state=random_state)),
                ("svm", SVC(kernel="rbf", C=10, probability=True,
                    class_weight="balanced", random_state=random_state)),
            ],
            voting="soft")
        return {
            "Random Forest": rf,
            "SVM (RBF)": svm,
            "Neural Network (MLP)": ann,
            "Gradient Boosting": gb,
            "Naive Bayes": nb,
            "AdaBoost-Voting Ensemble": voting,
        }

    models = get_models()

    print("\n" + "=" * 70)
    print("RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        fold_metrics = {
            "f1": [], "auc_pr": [], "mcc": [],
            "precision": [], "recall": [], "train_time": []
        }

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if IMBLEARN_AVAILABLE:
                smote_enn = SMOTEENN(random_state=random_state)
                try:
                    X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)
                except Exception:
                    X_train_res, y_train_res = X_train, y_train
            else:
                X_train_res, y_train_res = X_train, y_train

            start_time = time.time()
            model.fit(X_train_res, y_train_res)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred.astype(float)

            fold_metrics["f1"].append(f1_score(y_test, y_pred))
            fold_metrics["auc_pr"].append(average_precision_score(y_test, y_proba))
            fold_metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))
            fold_metrics["precision"].append(
                precision_score(y_test, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_test, y_pred))
            fold_metrics["train_time"].append(train_time)

            print(f"  Fold {fold_idx+1}: F1={fold_metrics['f1'][-1]:.4f}  "
                  f"AUC-PR={fold_metrics['auc_pr'][-1]:.4f}  "
                  f"MCC={fold_metrics['mcc'][-1]:.4f}")

        results[model_name] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in fold_metrics.items()
        }
        results[model_name]["category"] = (
            "Hybrid" if "Ensemble" in model_name else "Supervised")

        m = results[model_name]
        print(f"  MEAN: F1={m['f1']['mean']:.4f}+-{m['f1']['std']:.4f}  "
              f"AUC-PR={m['auc_pr']['mean']:.4f}+-{m['auc_pr']['std']:.4f}  "
              f"MCC={m['mcc']['mean']:.4f}+-{m['mcc']['std']:.4f}")

    # ---- K-Means (unsupervised) ----
    print(f"\n--- K-Means Clustering ---")
    kmeans = KMeans(n_clusters=10, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)
    sil_score = float(silhouette_score(X, clusters, sample_size=5000))

    cluster_distances = np.linalg.norm(
        kmeans.cluster_centers_ - X.mean(axis=0), axis=1)
    anomalous_clusters = np.argsort(cluster_distances)[-3:]
    anomalous_mask = np.isin(clusters, anomalous_clusters)
    fraud_in_anomalous = int(y[anomalous_mask].sum())
    total_fraud = int(y.sum())
    fraud_concentration = fraud_in_anomalous / total_fraud if total_fraud > 0 else 0

    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Fraud in anomalous clusters: {fraud_in_anomalous}/{total_fraud} "
          f"({fraud_concentration*100:.1f}%)")

    results["K-Means"] = {
        "category": "Unsupervised",
        "silhouette_score": sil_score,
        "fraud_concentration": float(fraud_concentration),
        "anomalous_cluster_pct": float(anomalous_mask.mean()),
    }

    # ---- AIS (Negative Selection — simplified) ----
    print(f"\n--- Artificial Immune System (Negative Selection) ---")
    legit_mask = y == 0
    legit_mean = X[legit_mask].mean(axis=0)
    legit_std = X[legit_mask].std(axis=0).clip(min=0.01)
    z_scores = np.abs((X - legit_mean) / legit_std).mean(axis=1)
    ais_threshold = np.percentile(z_scores[legit_mask], 90)
    ais_predictions = (z_scores > ais_threshold).astype(int)

    ais_recall = float(recall_score(y, ais_predictions))
    ais_precision = float(precision_score(y, ais_predictions, zero_division=0))
    ais_fpr = float(ais_predictions[y == 0].mean())

    print(f"  Coverage (recall): {ais_recall:.4f}")
    print(f"  Precision: {ais_precision:.4f}")
    print(f"  False positive rate: {ais_fpr:.4f}")

    results["AIS (Negative Selection)"] = {
        "category": "Unsupervised",
        "recall": ais_recall,
        "precision": ais_precision,
        "false_positive_rate": ais_fpr,
    }

    # ---- Feature Importance (Gradient Boosting) ----
    print(f"\n--- Feature Importance (Gradient Boosting) ---")
    gb_full = GradientBoostingClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=random_state)
    gb_full.fit(X, y)

    shap_info = {}
    try:
        import shap
        explainer = shap.TreeExplainer(gb_full)
        shap_values = explainer.shap_values(X[:1000])
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        print("  Top features by mean |SHAP|:")
        for rank, idx in enumerate(sorted_idx[:10]):
            print(f"    {rank+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")
        shap_info["feature_importance"] = {
            feature_names[i]: float(mean_abs_shap[i]) for i in sorted_idx[:10]
        }
        shap_info["method"] = "SHAP TreeExplainer"
    except ImportError:
        importances = gb_full.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("  Top features by Gini importance (SHAP not available):")
        for rank, idx in enumerate(sorted_idx[:10]):
            print(f"    {rank+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        shap_info["feature_importance"] = {
            feature_names[i]: float(importances[i]) for i in sorted_idx[:10]
        }
        shap_info["method"] = "Gini Importance (install shap for SHAP values)"

    return results, shap_info


# ============================================================
# 4. RESULTS EXPORT
# ============================================================

def export_results(results, shap_info, output_dir):
    """Save results as JSON and formatted markdown table."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "shap": shap_info,
                    "timestamp": datetime.now().isoformat()}, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Markdown table
    md_path = os.path.join(output_dir, "results_table.md")
    with open(md_path, "w") as f:
        f.write("# Experiment Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Classification Performance (5-Fold CV)\n\n")
        f.write("| Model | Category | F1-Score | AUC-PR | MCC | Precision | Recall |\n")
        f.write("|-------|----------|----------|--------|-----|-----------|--------|\n")

        supervised = {k: v for k, v in results.items()
                      if isinstance(v, dict)
                      and v.get("category") in ("Supervised", "Hybrid")}
        for name, metrics in sorted(supervised.items(),
                                     key=lambda x: x[1]["f1"]["mean"], reverse=True):
            f.write(
                f"| {name} | {metrics['category']} | "
                f"{metrics['f1']['mean']:.3f}+-{metrics['f1']['std']:.3f} | "
                f"{metrics['auc_pr']['mean']:.3f}+-{metrics['auc_pr']['std']:.3f} | "
                f"{metrics['mcc']['mean']:.3f}+-{metrics['mcc']['std']:.3f} | "
                f"{metrics['precision']['mean']:.3f}+-{metrics['precision']['std']:.3f} | "
                f"{metrics['recall']['mean']:.3f}+-{metrics['recall']['std']:.3f} |\n"
            )

        f.write("\n## Unsupervised Model Performance\n\n")
        if "K-Means" in results:
            km = results["K-Means"]
            f.write(f"**K-Means**: Silhouette={km['silhouette_score']:.3f}, "
                    f"Fraud concentration={km['fraud_concentration']*100:.1f}%\n\n")
        if "AIS (Negative Selection)" in results:
            ais = results["AIS (Negative Selection)"]
            f.write(f"**AIS**: Recall={ais['recall']:.3f}, "
                    f"Precision={ais['precision']:.3f}, "
                    f"FPR={ais['false_positive_rate']:.3f}\n\n")

        if shap_info.get("feature_importance"):
            f.write(f"## Feature Importance ({shap_info.get('method', 'N/A')})\n\n")
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for rank, (feat, imp) in enumerate(
                    shap_info["feature_importance"].items(), 1):
                f.write(f"| {rank} | {feat} | {imp:.4f} |\n")

    print(f"Results table saved to {md_path}")


# ============================================================
# 5. MAIN
# ============================================================

def run_pipeline(dataset_name, X, y, feature_names, output_dir):
    """Run the full experiment pipeline on a single dataset."""
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 70}")
    print(f"Feature matrix: {X.shape}, Features: {feature_names}")
    print(f"Class balance: {y.sum()} fraud / {len(y)} total "
          f"({y.mean()*100:.1f}%)")

    results, shap_info = run_experiments(X, y, feature_names)
    results["_dataset"] = dataset_name

    sub_dir = os.path.join(output_dir, dataset_name.lower().replace(" ", "_"))
    export_results(results, shap_info, sub_dir)

    return results, shap_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HealthFraudML — ML Comparison for Healthcare Fraud Detection")
    parser.add_argument("--kaggle", action="store_true",
                        help="Use Kaggle Healthcare Provider Fraud dataset")
    parser.add_argument("--both", action="store_true",
                        help="Run on both synthetic AND Kaggle datasets")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing Kaggle CSV files "
                             "(default: ../data/kaggle/)")
    args = parser.parse_args()

    print("=" * 70)
    print("HealthFraudML - Comparative ML Analysis for Healthcare Fraud Detection")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    run_synthetic = not args.kaggle or args.both
    run_kaggle = args.kaggle or args.both

    all_results = {}

    # ---- Synthetic dataset ----
    if run_synthetic:
        print("\n[SYNTHETIC] Generating 50K healthcare claims dataset...")
        df_synth = generate_synthetic_claims(n_samples=50000, fraud_rate=0.032)

        csv_path = os.path.join(output_dir, "synthetic_claims.csv")
        df_synth.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")

        X_s, y_s, feat_s, _ = preprocess(df_synth)
        results_s, shap_s = run_pipeline("Synthetic", X_s, y_s, feat_s, output_dir)
        all_results["synthetic"] = results_s

    # ---- Kaggle dataset ----
    if run_kaggle:
        data_dir = args.data_dir or os.path.join(base_dir, "..", "data", "kaggle")
        try:
            df_kaggle = load_kaggle_dataset(data_dir)
            X_k, y_k, feat_k, _ = preprocess_kaggle(df_kaggle)
            results_k, shap_k = run_pipeline("Kaggle_Medicare", X_k, y_k,
                                              feat_k, output_dir)
            all_results["kaggle"] = results_k
        except FileNotFoundError as e:
            print(f"\n[KAGGLE] SKIPPED — {e}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    if run_synthetic:
        print(f"  - synthetic/experiment_results.json")
        print(f"  - synthetic/results_table.md")
    if run_kaggle:
        print(f"  - kaggle_medicare/experiment_results.json")
        print(f"  - kaggle_medicare/results_table.md")
    print(f"\nNext: Update paper with actual results from the results_table.md files")
