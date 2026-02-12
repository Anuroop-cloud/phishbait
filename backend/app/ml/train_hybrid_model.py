import os

# Resolve project root dynamically
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("BASE_DIR:", BASE_DIR)
print("DATASET_DIR:", DATASET_DIR)
print("MODEL_DIR:", MODEL_DIR)

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Paths
    train_path = os.path.join(DATASET_DIR, "train_processed.csv")
    test_path = os.path.join(DATASET_DIR, "test_processed.csv")

    # Load data
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Labels
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # ──────────────────────────────────────────
    # Load precomputed IndicBERT embeddings
    # ──────────────────────────────────────────
    print("Looking for embeddings at:")
    print(os.path.join(DATASET_DIR, "train_embeddings.npy"))

    X_train_embed = np.load(os.path.join(DATASET_DIR, "train_embeddings.npy"))
    X_test_embed = np.load(os.path.join(DATASET_DIR, "test_embeddings.npy"))

    print("Embedding shape:", X_train_embed.shape)
    assert X_train_embed.shape[1] == 768, "Embedding dimension must be 768"

    # ──────────────────────────────────────────
    # Load handcrafted features
    # ──────────────────────────────────────────
    feature_cols = [
        "url_count",
        "dot_count",
        "has_at_symbol",
        "urgency_flag",
        "threat_flag",
        "suspicious_domain_flag",
        "caps_word_count",
        "digit_count",
        "special_char_count",
    ]

    X_train_hand = train_df[feature_cols].values
    X_test_hand = test_df[feature_cols].values

    assert X_train_hand.shape[1] == 9, "Handcrafted features must be 9"

    # ──────────────────────────────────────────
    # Concatenate embeddings + handcrafted
    # ──────────────────────────────────────────
    X_train = np.concatenate([X_train_embed, X_train_hand], axis=1)
    X_test = np.concatenate([X_test_embed, X_test_hand], axis=1)

    print("Final feature dimension:", X_train.shape[1])
    assert X_train.shape[1] == 777, "Final dimension must be 777"

    # ──────────────────────────────────────────
    # Train model
    # ──────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("Training RandomForest classifier...")
    model.fit(X_train, y_train)

    # ──────────────────────────────────────────
    # Validate model feature count
    # ──────────────────────────────────────────
    print("Model expects features:", model.n_features_in_)
    assert model.n_features_in_ == 777, "Model feature count mismatch"

    # ──────────────────────────────────────────
    # Evaluate
    # ──────────────────────────────────────────
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    # ──────────────────────────────────────────
    # Save metrics
    # ──────────────────────────────────────────
    metrics = {
        "accuracy": float(acc),
        "feature_dimension": int(model.n_features_in_),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model metrics saved successfully.")

    # ──────────────────────────────────────────
    # Save model
    # ──────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "hybrid_random_forest.pkl")
    joblib.dump(model, model_path)
    print("Hybrid 777-d model saved successfully.")


if __name__ == "__main__":
    main()
