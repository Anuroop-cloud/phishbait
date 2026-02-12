"""
system_integrity_check.py

Validate end-to-end compatibility between every component of the
multilingual phishing detection pipeline:

  1. Processed train / test CSV files
  2. Handcrafted features  (9 columns)
  3. IndicBERT embeddings  (768-d)
  4. Hybrid RandomForest model
  5. Single-sample prediction
  6. Batch prediction

Raises clear errors on any mismatch.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os

import joblib
import numpy as np
import pandas as pd

from data_pipeline import DataPipeline
from indicbert_encoder import IndicBERTEncoder

# Expected feature layout
FEATURE_COLS = [
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

EMBED_DIM = 768
EXPECTED_DIM = EMBED_DIM + len(FEATURE_COLS)  # 777

MODEL_PATH = "models/hybrid_random_forest.pkl"
TRAIN_CSV = "dataset/train_processed.csv"
TEST_CSV = "dataset/test_processed.csv"
TRAIN_EMB = "dataset/train_embeddings.npy"
TEST_EMB = "dataset/test_embeddings.npy"


def _section(title: str) -> None:
    """Pretty-print a section header."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ──────────────────────────────────────────────
# 2. LOAD PROCESSED DATA
# ──────────────────────────────────────────────
def check_processed_data():
    """Load and validate train / test CSV files."""
    _section("2. LOAD PROCESSED DATA")

    for path in [TRAIN_CSV, TEST_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing processed file: {path}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"Train shape : {train_df.shape}")
    print(f"Test  shape : {test_df.shape}")
    print(f"Columns     : {list(train_df.columns)}")

    return train_df, test_df


# ──────────────────────────────────────────────
# 3. VERIFY HANDCRAFTED FEATURES
# ──────────────────────────────────────────────
def check_handcrafted_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Ensure all 9 handcrafted feature columns exist."""
    _section("3. VERIFY HANDCRAFTED FEATURES")

    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing handcrafted feature columns in train: {missing}")

    missing_test = [c for c in FEATURE_COLS if c not in test_df.columns]
    if missing_test:
        raise ValueError(f"Missing handcrafted feature columns in test: {missing_test}")

    print(f"All {len(FEATURE_COLS)} handcrafted feature columns present ✓")
    print(f"Columns: {FEATURE_COLS}")


# ──────────────────────────────────────────────
# 4. VERIFY SAVED EMBEDDINGS
# ──────────────────────────────────────────────
def check_embeddings(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Load .npy embeddings and validate shapes."""
    _section("4. VERIFY SAVED EMBEDDINGS")

    for path in [TRAIN_EMB, TEST_EMB]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing embedding file: {path}")

    train_emb = np.load(TRAIN_EMB)
    test_emb = np.load(TEST_EMB)

    print(f"train_embeddings shape : {train_emb.shape}")
    print(f"test_embeddings  shape : {test_emb.shape}")

    # Check embedding dimension
    if train_emb.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Train embedding dim is {train_emb.shape[1]}, expected {EMBED_DIM}"
        )
    if test_emb.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Test embedding dim is {test_emb.shape[1]}, expected {EMBED_DIM}"
        )

    # Check row counts match
    if train_emb.shape[0] != len(train_df):
        raise ValueError(
            f"Train embedding rows ({train_emb.shape[0]}) != "
            f"train CSV rows ({len(train_df)})"
        )
    if test_emb.shape[0] != len(test_df):
        raise ValueError(
            f"Test embedding rows ({test_emb.shape[0]}) != "
            f"test CSV rows ({len(test_df)})"
        )

    print(f"Embedding dimension == {EMBED_DIM} ✓")
    print("Row counts match CSV files ✓")

    return train_emb, test_emb


# ──────────────────────────────────────────────
# 5. VERIFY CONCATENATION
# ──────────────────────────────────────────────
def check_concatenation(train_df: pd.DataFrame, train_emb: np.ndarray):
    """Concatenate embeddings + handcrafted features and validate final dim."""
    _section("5. VERIFY CONCATENATION")

    X_train_hand = train_df[FEATURE_COLS].values
    X_train = np.concatenate([train_emb, X_train_hand], axis=1)

    print(f"Handcrafted features shape : {X_train_hand.shape}")
    print(f"Embedding shape            : {train_emb.shape}")
    print(f"Concatenated shape         : {X_train.shape}")
    print(f"Final dimension            : {X_train.shape[1]}")

    if X_train.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Concatenated dimension is {X_train.shape[1]}, "
            f"expected {EXPECTED_DIM}"
        )

    print(f"Final dimension == {EXPECTED_DIM} ✓")
    return X_train


# ──────────────────────────────────────────────
# 6. VERIFY MODEL LOAD
# ──────────────────────────────────────────────
def check_model():
    """Load the hybrid RandomForest model and validate its properties."""
    _section("6. VERIFY MODEL LOAD")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print(f"Model type              : {type(model).__name__}")
    print(f"n_estimators            : {model.n_estimators}")
    print(f"feature_importances len : {len(model.feature_importances_)}")

    if len(model.feature_importances_) != EXPECTED_DIM:
        raise ValueError(
            f"Model expects {len(model.feature_importances_)} features, "
            f"but pipeline produces {EXPECTED_DIM}"
        )

    print(f"Feature importances length == {EXPECTED_DIM} ✓")
    return model


# ──────────────────────────────────────────────
# 7. TEST SINGLE PREDICTION
# ──────────────────────────────────────────────
def check_single_prediction(model, encoder: IndicBERTEncoder):
    """Run a single end-to-end prediction."""
    _section("7. TEST SINGLE PREDICTION")

    sample_text = (
        "Your account will be suspended immediately. "
        "Click http://bit.ly/abc"
    )

    # Generate embedding (768,)
    embedding = encoder.encode(sample_text)
    print(f"Embedding shape: {embedding.shape}")

    # Extract handcrafted features (9,)
    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.__init__("dummy")  # only need keyword lists
    features = pipeline.extract_features(sample_text)
    print(f"Handcrafted features shape: {features.shape}")

    # Concatenate → (777,)
    combined = np.concatenate([embedding, features])
    print(f"Combined vector shape: {combined.shape}")

    # Reshape for model → (1, 777)
    X = combined.reshape(1, -1)

    # Predict
    prediction = model.predict(X)
    probability = model.predict_proba(X)

    print(f"\nSample text : \"{sample_text}\"")
    print(f"Prediction  : {prediction[0]}")
    print(f"Probability : {probability[0]}")

    # Top 5 important features
    importances = model.feature_importances_
    feature_names = [f"emb_{i}" for i in range(EMBED_DIM)] + FEATURE_COLS
    top_idx = np.argsort(importances)[::-1][:5]

    print("\nTop 5 important features:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {feature_names[idx]:30s}  importance={importances[idx]:.6f}")


# ──────────────────────────────────────────────
# 8. TEST BATCH PREDICTION
# ──────────────────────────────────────────────
def check_batch_prediction(
    model, encoder: IndicBERTEncoder, test_df: pd.DataFrame
):
    """Run batch prediction on 5 test samples."""
    _section("8. TEST BATCH PREDICTION")

    sample_df = test_df.head(5).copy()
    texts = sample_df["text"].fillna("").astype(str).tolist()

    # Encode batch
    embeddings = encoder.encode_batch(texts, batch_size=16)
    print(f"Batch embeddings shape: {embeddings.shape}")

    # Extract handcrafted features
    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline.__init__("dummy")
    hand_features = np.array([pipeline.extract_features(t) for t in texts])
    print(f"Batch handcrafted shape: {hand_features.shape}")

    # Concatenate
    X = np.concatenate([embeddings, hand_features], axis=1)
    print(f"Batch combined shape: {X.shape}")

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print("\nBatch predictions:")
    for i in range(len(texts)):
        label = predictions[i]
        prob = probabilities[i]
        snippet = texts[i][:60] + ("..." if len(texts[i]) > 60 else "")
        print(f"  [{i+1}] pred={label}  prob={prob}  text=\"{snippet}\"")


# ──────────────────────────────────────────────
# 9. MAIN — RUN ALL CHECKS
# ──────────────────────────────────────────────
def main() -> None:
    """Execute every integrity check in sequence."""
    _section("SYSTEM INTEGRITY CHECK — START")

    # 2. Processed data
    train_df, test_df = check_processed_data()

    # 3. Handcrafted features
    check_handcrafted_features(train_df, test_df)

    # 4. Saved embeddings
    train_emb, test_emb = check_embeddings(train_df, test_df)

    # 5. Concatenation
    check_concatenation(train_df, train_emb)

    # 6. Model
    model = check_model()

    # 7 & 8 require the encoder — load once
    print("\n[Integrity] Initializing IndicBERT encoder for live tests...")
    encoder = IndicBERTEncoder()

    # 7. Single prediction
    check_single_prediction(model, encoder)

    # 8. Batch prediction
    check_batch_prediction(model, encoder, test_df)

    # ── All passed ──
    _section("9. FINAL RESULT")
    print("✅  SYSTEM INTEGRITY CHECK PASSED")
    print(f"    Embedding dim    : {EMBED_DIM}")
    print(f"    Handcrafted feats: {len(FEATURE_COLS)}")
    print(f"    Total features   : {EXPECTED_DIM}")
    print(f"    Model            : {MODEL_PATH}")


if __name__ == "__main__":
    main()
