"""
train_url_model.py

Train a RandomForest classifier on the PhiUSIIL Kaggle phishing URL dataset.

Uses 20 structural URL features extracted from the raw URL string
(matching url_feature_extractor.py). The Kaggle dataset already provides
many of these columns, so we map them directly.

Saves:
  models/url_random_forest.pkl
  models/url_model_metrics.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

KAGGLE_CSV = os.path.join(DATASET_DIR, "phishing_url_dataset.csv")

# ── The 20 features we'll use (subset of Kaggle columns, mapped) ─
# We pick the columns from the Kaggle dataset that correspond to
# our url_feature_extractor.py features. This ensures the trained
# model can score features extracted at inference time.
KAGGLE_TO_OUR_MAPPING = {
    "URLLength":                  "url_length",
    "DomainLength":               "domain_length",
    "IsHTTPS":                    "is_https",
    "IsDomainIP":                 "is_domain_ip",
    "NoOfSubDomain":              "num_subdomains",
    "TLDLength":                  "tld_length",
    "NoOfDegitsInURL":            "num_digits_in_url",
    "DegitRatioInURL":            "digit_ratio",
    "LetterRatioInURL":           "letter_ratio",
    "NoOfOtherSpecialCharsInURL": "num_special_chars",
    "SpacialCharRatioInURL":      "special_char_ratio",
    "HasObfuscation":             "has_obfuscation",
    "NoOfLettersInURL":           "num_letters_in_url",
    "ObfuscationRatio":           "obfuscation_ratio",
    "CharContinuationRate":       "char_continuation_rate",
    "URLSimilarityIndex":         "url_similarity_index",
    "URLCharProb":                "url_char_prob",
    "TLDLegitimateProb":          "tld_legitimate_prob",
    "NoOfEqualsInURL":            "num_equals",
    "NoOfQMarkInURL":             "num_qmarks",
}

# The 20 columns in OUR feature order (used at inference time)
URL_FEATURE_COLS_KAGGLE = list(KAGGLE_TO_OUR_MAPPING.keys())


def load_and_prepare() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Kaggle CSV, select features, split."""
    print(f"[url_train] Loading dataset from {KAGGLE_CSV} ...")
    df = pd.read_csv(KAGGLE_CSV)
    print(f"[url_train] Raw shape: {df.shape}")

    # Keep only our feature columns + label
    missing = [c for c in URL_FEATURE_COLS_KAGGLE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Kaggle CSV: {missing}")

    X = df[URL_FEATURE_COLS_KAGGLE].values.astype(np.float64)
    y = df["label"].values.astype(int)

    # Handle NaN / inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[url_train] Feature matrix: {X.shape}")
    print(f"[url_train] Label dist: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    print(f"[url_train] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_and_save() -> None:
    """Train RandomForest and persist model + metrics."""
    X_train, X_test, y_train, y_test = load_and_prepare()

    print("[url_train] Training RandomForest (n_estimators=200) ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[url_train] URL Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    # Save model
    model_path = os.path.join(MODEL_DIR, "url_random_forest.pkl")
    joblib.dump(clf, model_path)
    print(f"[url_train] Model saved → {model_path}")

    # Save metrics
    metrics = {
        "accuracy": acc,
        "feature_count": X_train.shape[1],
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "feature_columns_kaggle": URL_FEATURE_COLS_KAGGLE,
    }
    metrics_path = os.path.join(MODEL_DIR, "url_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[url_train] Metrics saved → {metrics_path}")


if __name__ == "__main__":
    train_and_save()
