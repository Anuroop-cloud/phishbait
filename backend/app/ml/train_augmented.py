"""
train_augmented.py

End-to-end training pipeline for the augmented phishing dataset.

Steps:
  1. Load & deduplicate the augmented CSV
  2. Clean text + extract 9 handcrafted features
  3. Stratified 80/20 split
  4. Generate frozen IndicBERT CLS embeddings (batched)
  5. Concatenate embeddings + handcrafted features → 777-d vectors
  6. Train RandomForest with class_weight='balanced'
  7. Evaluate: accuracy, precision, recall, F1, classification report
  8. Save model (.pkl), processed CSVs, embeddings, metrics JSON

Usage:
    python backend/app/ml/train_augmented.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Resolve paths ────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from indicbert_encoder import IndicBERTEncoder
from data_pipeline import DataPipeline


# ── Config ───────────────────────────────────────────────
AUGMENTED_CSV = os.path.join(DATASET_DIR, "phishing_dataset_large_multilingual.csv")
TRAIN_CSV_OUT = os.path.join(DATASET_DIR, "train_processed.csv")
TEST_CSV_OUT = os.path.join(DATASET_DIR, "test_processed.csv")
TRAIN_EMB_OUT = os.path.join(DATASET_DIR, "train_embeddings.npy")
TEST_EMB_OUT = os.path.join(DATASET_DIR, "test_embeddings.npy")
MODEL_OUT = os.path.join(MODEL_DIR, "hybrid_random_forest.pkl")
METRICS_OUT = os.path.join(MODEL_DIR, "model_metrics.json")

FEATURE_COLUMNS = [
    "url_count", "dot_count", "has_at_symbol",
    "urgency_flag", "threat_flag", "suspicious_domain_flag",
    "caps_word_count", "digit_count", "special_char_count",
]


def main() -> None:
    t_start = time.time()

    # ================================================================
    # 1. LOAD & DEDUPLICATE
    # ================================================================
    print("=" * 60)
    print("STEP 1: Loading & deduplicating dataset")
    print("=" * 60)

    df = pd.read_csv(AUGMENTED_CSV)
    print(f"  Raw rows        : {len(df)}")

    # Ensure columns exist
    assert "text" in df.columns and "label" in df.columns, \
        f"Expected 'text' and 'label' columns, got {list(df.columns)}"

    # Drop nulls
    df = df.dropna(subset=["text", "label"])

    # Deduplicate on text
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  After dedup     : {len(df)}  (dropped {before - len(df)} duplicates)")

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Class distribution
    counts = df["label"].value_counts().to_dict()
    print(f"  Class 0 (legit) : {counts.get(0, 0)}")
    print(f"  Class 1 (phish) : {counts.get(1, 0)}")

    # ================================================================
    # 2. CLEAN TEXT & EXTRACT HANDCRAFTED FEATURES
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Text cleaning & feature extraction")
    print("=" * 60)

    # Use DataPipeline for cleaning and features
    # We instantiate with a dummy path since we're providing df directly
    pipeline = DataPipeline(AUGMENTED_CSV)

    # Clean text — preserve original for caps detection
    df["clean_text"] = df["text"].apply(pipeline.clean_text)

    # Extract 9 handcrafted features from ORIGINAL text (preserves case)
    features = df["text"].apply(pipeline.extract_features)
    feature_matrix = np.stack(features.values)
    for idx, col_name in enumerate(FEATURE_COLUMNS):
        df[col_name] = feature_matrix[:, idx]

    print(f"  Clean text sample: '{df['clean_text'].iloc[0][:80]}...'")
    print(f"  Features added  : {len(FEATURE_COLUMNS)}")

    # ================================================================
    # 3. STRATIFIED 80/20 SPLIT
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Stratified train/test split")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"  Train samples   : {len(train_df)}")
    print(f"  Test samples    : {len(test_df)}")
    print(f"  Train label dist: {train_df['label'].value_counts().to_dict()}")
    print(f"  Test  label dist: {test_df['label'].value_counts().to_dict()}")

    # Save processed CSVs
    train_df.to_csv(TRAIN_CSV_OUT, index=False)
    test_df.to_csv(TEST_CSV_OUT, index=False)
    print(f"  Saved: {TRAIN_CSV_OUT}")
    print(f"  Saved: {TEST_CSV_OUT}")

    # ================================================================
    # 4. GENERATE INDICBERT EMBEDDINGS (BATCHED)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Generating IndicBERT embeddings (frozen, batched)")
    print("=" * 60)

    encoder = IndicBERTEncoder()

    train_texts = train_df["clean_text"].fillna("").astype(str).tolist()
    test_texts = test_df["clean_text"].fillna("").astype(str).tolist()

    t_embed_start = time.time()
    print(f"  Encoding {len(train_texts)} train texts (batch_size=32) ...")
    train_embeddings = encoder.encode_batch(train_texts, batch_size=32)

    print(f"  Encoding {len(test_texts)} test texts (batch_size=32) ...")
    test_embeddings = encoder.encode_batch(test_texts, batch_size=32)
    t_embed = time.time() - t_embed_start

    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Test  embeddings: {test_embeddings.shape}")
    print(f"  Embedding time  : {t_embed:.1f}s")

    # Save embeddings
    np.save(TRAIN_EMB_OUT, train_embeddings)
    np.save(TEST_EMB_OUT, test_embeddings)
    print(f"  Saved: {TRAIN_EMB_OUT}")
    print(f"  Saved: {TEST_EMB_OUT}")

    # ================================================================
    # 5. BUILD FINAL FEATURE MATRIX (777-d)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Building 777-d feature vectors")
    print("=" * 60)

    X_train_hand = train_df[FEATURE_COLUMNS].values.astype(np.float64)
    X_test_hand = test_df[FEATURE_COLUMNS].values.astype(np.float64)

    X_train = np.concatenate([train_embeddings, X_train_hand], axis=1)
    X_test = np.concatenate([test_embeddings, X_test_hand], axis=1)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test : {X_test.shape}")
    assert X_train.shape[1] == 777, f"Expected 777-d, got {X_train.shape[1]}"

    # ================================================================
    # 6. TRAIN RANDOM FOREST
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Training RandomForest classifier")
    print("=" * 60)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,        # let trees grow fully
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    t_train_start = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t_train_start
    print(f"  Training time   : {t_train:.1f}s")
    print(f"  Model features  : {clf.n_features_in_}")

    # ================================================================
    # 7. EVALUATE
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Evaluation")
    print("=" * 60)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Phishing"]
    )

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification Report:\n{report}")

    # Quick 5-fold CV on training data for stability check
    print("  Running 5-fold cross-validation on train set ...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    print(f"  CV F1 scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV F1 mean   : {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # ================================================================
    # 8. SAVE MODEL & METRICS
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Saving model & metrics")
    print("=" * 60)

    joblib.dump(clf, MODEL_OUT)
    print(f"  Model saved : {MODEL_OUT}")

    t_total = time.time() - t_start
    metrics = {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "cv_f1_mean": round(float(cv_scores.mean()), 6),
        "cv_f1_std": round(float(cv_scores.std()), 6),
        "feature_dimension": int(X_train.shape[1]),
        "embedding_dimension": int(train_embeddings.shape[1]),
        "handcrafted_features": len(FEATURE_COLUMNS),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "total_unique_samples": int(len(df)),
        "class_distribution": {
            "train": {
                "legitimate": int((y_train == 0).sum()),
                "phishing": int((y_train == 1).sum()),
            },
            "test": {
                "legitimate": int((y_test == 0).sum()),
                "phishing": int((y_test == 1).sum()),
            },
        },
        "confusion_matrix": {
            "TN": int(cm[0][0]),
            "FP": int(cm[0][1]),
            "FN": int(cm[1][0]),
            "TP": int(cm[1][1]),
        },
        "embedding_time_seconds": round(t_embed, 1),
        "training_time_seconds": round(t_train, 1),
        "total_time_seconds": round(t_total, 1),
        "model_path": MODEL_OUT,
        "encoder": "ai4bharat/IndicBERTv2-MLM-only",
        "classifier": "RandomForest(n_estimators=200, class_weight=balanced)",
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {METRICS_OUT}")

    # ================================================================
    # TRAINING SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Dataset           : phishing_dataset_augmented.csv")
    print(f"  Unique samples    : {len(df)}")
    print(f"  Train / Test      : {len(y_train)} / {len(y_test)}")
    print(f"  Embedding dim     : {train_embeddings.shape[1]}")
    print(f"  Feature dim       : {X_train.shape[1]} (768 IndicBERT + 9 handcrafted)")
    print(f"  Embedding time    : {t_embed:.1f}s")
    print(f"  Training time     : {t_train:.1f}s")
    print(f"  Total time        : {t_total:.1f}s")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  CV F1 (5-fold)    : {cv_scores.mean():.4f} +/- {cv_scores.std()*2:.4f}")
    print(f"  Model saved       : {MODEL_OUT}")
    print(f"  Metrics saved     : {METRICS_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
