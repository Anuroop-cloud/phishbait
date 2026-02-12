"""
inference.py

Production-ready single-text prediction for the multilingual
phishing detection system.

Components loaded once at module level:
  - Hybrid RandomForest model   (models/hybrid_random_forest.pkl)
  - IndicBERT encoder           (768-d CLS embeddings)
  - DataPipeline                (text cleaning + 9 handcrafted features)

Final feature vector: 768 (embedding) + 9 (handcrafted) = 777 dimensions.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import json
import os
import sys

import numpy as np
import joblib

# Resolve project root dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Ensure local imports work regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from indicbert_encoder import IndicBERTEncoder
from data_pipeline import DataPipeline


# ──────────────────────────────────────────────
# 2. LOAD MODEL & ENCODER ONCE (GLOBAL)
# ──────────────────────────────────────────────
print("[inference] Loading hybrid RandomForest model...")
model = joblib.load(os.path.join(MODEL_DIR, "hybrid_random_forest.pkl"))

print("[inference] Initializing IndicBERT encoder...")
encoder = IndicBERTEncoder()

# Only need the pipeline instance for clean_text() and extract_features()
pipeline = DataPipeline(os.path.join(DATASET_DIR, "train_processed.csv"))

# Load saved model metrics
metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        MODEL_METRICS = json.load(f)
    print(f"[inference] Model accuracy: {MODEL_METRICS.get('accuracy')}")
else:
    MODEL_METRICS = {"accuracy": None}
    print("[inference] WARNING: model_metrics.json not found.")

print("Model and encoder successfully initialized.")

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


# ──────────────────────────────────────────────
# 3. PREDICTION FUNCTION
# ──────────────────────────────────────────────
def predict(text: str) -> dict:
    """Run end-to-end phishing prediction on a single text.

    Parameters
    ----------
    text : str
        Raw input message (any supported language).

    Returns
    -------
    dict
        {
            "prediction": "Phishing" | "Safe",
            "confidence": float   # 0.0 – 1.0
        }
    """
    # Clean text
    clean = pipeline.clean_text(text)

    # Extract 9 handcrafted features
    features = pipeline.extract_features(clean)

    # Get 768-d IndicBERT embedding
    embedding = encoder.encode(clean)

    # Concatenate → (777,)
    final_vector = np.concatenate([embedding, features])

    # Reshape for sklearn → (1, 777)
    final_vector = final_vector.reshape(1, -1)

    # Predict
    pred = model.predict(final_vector)[0]
    prob = model.predict_proba(final_vector)[0]

    confidence = float(np.max(prob))
    label = "Phishing" if pred == 1 else "Safe"

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "model_accuracy": MODEL_METRICS.get("accuracy"),
    }


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Your account will be suspended immediately. Click http://bit.ly/abc",
        "Meeting scheduled for tomorrow at 10 AM.",
        "तुरंत अपना खाता सत्यापित करें। http://tinyurl.com/xyz",
    ]

    for text in samples:
        result = predict(text)
        snippet = text[:70] + ("..." if len(text) > 70 else "")
        print(f"Text : \"{snippet}\"")
        print(f"  → {result['prediction']}  (confidence {result['confidence']})\n")
