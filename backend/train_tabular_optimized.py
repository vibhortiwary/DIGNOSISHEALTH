# backend/train_tabular_optimized.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import joblib
import os

MODEL_DIR = "backend/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# Breast Cancer preprocessing
# -------------------------------------------------------
def preprocess_breast(df):
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Map diagnosis: M = malignant(1), B = benign(0)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


# -------------------------------------------------------
# Generic trainer (HEART or other simple tabular tasks)
# -------------------------------------------------------
def train_model(csv_path, target_col, model_name, preprocess_func=None):
    df = pd.read_csv(csv_path)

    if preprocess_func:
        df = preprocess_func(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # numeric only
    X = X.select_dtypes(include=["number"])
    feature_cols = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )

    is_multiclass = len(np.unique(y_train)) > 2

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss" if is_multiclass else "logloss",
        objective="multi:softprob" if is_multiclass else "binary:logistic",
        num_class=len(np.unique(y_train)) if is_multiclass else None,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print(f"\n📌 Metrics for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, pred))

    if is_multiclass:
        print("F1 Score:", f1_score(y_test, pred, average="weighted"))
        print("ROC-AUC:", roc_auc_score(y_test, prob, multi_class="ovo"))
    else:
        print("F1 Score:", f1_score(y_test, pred))
        print("ROC-AUC:", roc_auc_score(y_test, prob[:, 1]))

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "feature_names": feature_cols,  # safe for all
        },
        f"{MODEL_DIR}/{model_name}.pkl",
    )

    print(f"🔥 {model_name} saved!")


# -------------------------------------------------------
# SPECIAL: Diabetes trainer using NHANES/BRFSS-style features
# -------------------------------------------------------
def train_diabetes_model(csv_path, model_name):
    """
    Uses a compact feature set from diabeties_data.csv:

        Sex          (0 = Female, 1 = Male)
        HighBP       (0/1)
        HighChol     (0/1)
        Smoker       (0/1)
        PhysActivity (0/1)
        GenHlth      (1 = Excellent ... 5 = Poor)
        MentHlth     (0–30 days)
        BMI          (body mass index)
        Age          (1–13 BRFSS age code)

    Target:
        Diabetes_012 (0 = Normal, 1 = Prediabetes, 2 = Diabetes)
    """
    df = pd.read_csv(csv_path)

    feature_cols = [
        "Sex",
        "HighBP",
        "HighChol",
        "Smoker",
        "PhysActivity",
        "GenHlth",
        "MentHlth",
        "BMI",
        "Age",
    ]
    target_col = "Diabetes_012"

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"❌ diabeties_data.csv is missing columns: {missing}")

    df = df[feature_cols + [target_col]].copy()

    y = df[target_col]
    X = df[feature_cols]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print(f"\n📌 Metrics for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, average="weighted"))
    print("ROC-AUC:", roc_auc_score(y_test, prob, multi_class="ovo"))

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "feature_names": feature_cols,
        },
        f"{MODEL_DIR}/{model_name}.pkl",
    )

    print(f"🔥 {model_name} saved!")


# -------------------------------------------------------
# SPECIAL: Breast trainer  ✅ OPTION A (ONLY 10 UI FEATURES)
# -------------------------------------------------------
def train_breast_model(csv_path, model_name):
    """
    Train on ONLY the 10 features that your frontend + normal_ranges.json use:

        radius_mean
        texture_mean
        perimeter_mean
        area_mean
        smoothness_mean
        compactness_mean
        concavity_mean
        symmetry_mean
        fractal_dimension_mean
        radius_worst
    """
    df = pd.read_csv(csv_path)
    df = preprocess_breast(df)

    # target
    y = df["diagnosis"]

    # exactly the same 10 features as UI + normal_ranges.json
    feature_cols = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_worst",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ breast_data.csv is missing columns: {missing}")

    X = df[feature_cols]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        objective="binary:logistic",
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print(f"\n📌 Metrics for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, prob[:, 1]))

    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "feature_names": feature_cols,  # IMPORTANT: matches frontend
        },
        f"{MODEL_DIR}/{model_name}.pkl",
    )

    print(f"🔥 {model_name} saved!")


# -------------------------------------------------------
# TRAIN ALL MODELS
# -------------------------------------------------------

# HEART (multiclass target = num)
train_model("backend/models/heart_data.csv", "num", "heart_optimal")

# DIABETES (3-class 0/1/2 using compact NHANES-style features)
train_diabetes_model("backend/models/diabeties_data.csv", "diabetes_optimal")

# BREAST (binary target = diagnosis) – trained on ONLY the 10 UI features
train_breast_model("backend/models/breast_data.csv", "breast_optimal")
