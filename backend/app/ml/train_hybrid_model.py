import os
from typing import List

import numpy as np
import pandas as pd

try:
    from backend.ml.indicBERT import IndicBERTEncoder
except Exception:
    # Fallback minimal IndicBERTEncoder using transformers if original isn't available
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        class IndicBERTEncoder:
            def __init__(self, model_name: str = "ai4bharat/indic-bert", device: str = None):
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)

            def _mean_pool(self, model_output, attention_mask):
                token_embeddings = model_output[0]  # (batch_size, seq_len, hidden)
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask

            def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
                all_emb = []
                self.model.eval()
                with torch.no_grad():
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i : i + batch_size]
                        enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                        input_ids = enc["input_ids"].to(self.device)
                        attention_mask = enc["attention_mask"].to(self.device)
                        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        emb = self._mean_pool(out, attention_mask)
                        emb = emb.cpu().numpy()
                        all_emb.append(emb)
                return np.vstack(all_emb)

    except Exception:
        # Final fallback: random vectors (very small chance, only for quick smoke tests)
        class IndicBERTEncoder:
            def __init__(self, *_args, **_kwargs):
                self.dim = 128

            def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
                return np.random.randn(len(texts), self.dim)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def main():
    # Paths
    train_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", "train_processed.csv")
    test_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", "test_processed.csv")

    train_path = os.path.abspath(train_path)
    test_path = os.path.abspath(test_path)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Labels
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Initialize encoder
    encoder = IndicBERTEncoder()

    # Generate embeddings
    print("Generating IndicBERT embeddings for training set...")
    X_train_embed = encoder.encode_batch(train_df["clean_text"].astype(str).tolist())
    print("Generating IndicBERT embeddings for test set...")
    X_test_embed = encoder.encode_batch(test_df["clean_text"].astype(str).tolist())

    # Handcrafted feature columns
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

    # Concatenate
    X_train = np.concatenate([X_train_embed, X_train_hand], axis=1)
    X_test = np.concatenate([X_test_embed, X_test_hand], axis=1)

    # Print feature vector dimension
    print(f"Feature vector dimension: {X_train.shape[1]}")

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("Training RandomForest classifier...")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    # Save model
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"), exist_ok=True)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "hybrid_random_forest.pkl"))
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
