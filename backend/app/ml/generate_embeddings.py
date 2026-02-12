"""
generate_embeddings.py

Generate and save IndicBERT embeddings for processed train/test data.

Input files:
  - dataset/train_processed.csv
  - dataset/test_processed.csv

Output files:
  - dataset/train_embeddings.npy
  - dataset/test_embeddings.npy

Runs on CPU only. No fine-tuning.
"""

import os
import time

import numpy as np
import pandas as pd

from indicbert_encoder import IndicBERTEncoder


def main() -> None:
    """Generate and persist IndicBERT embeddings for train and test splits."""
    start_time = time.time()

    train_path = "dataset/train_processed.csv"
    test_path = "dataset/test_processed.csv"

    # 1) Load processed train/test CSV files
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    print("[generate_embeddings] Loading processed datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "clean_text" not in train_df.columns or "clean_text" not in test_df.columns:
        raise ValueError("Both CSV files must contain a 'clean_text' column.")

    # Convert to strings and handle nulls safely
    train_texts = train_df["clean_text"].fillna("").astype(str).tolist()
    test_texts = test_df["clean_text"].fillna("").astype(str).tolist()

    print(f"[generate_embeddings] Train samples: {len(train_texts)}")
    print(f"[generate_embeddings] Test samples : {len(test_texts)}")

    # 2) Initialize encoder
    print("[generate_embeddings] Initializing IndicBERT encoder on CPU...")
    encoder = IndicBERTEncoder()

    # 3) Encode with batch_size=16
    print("[generate_embeddings] Encoding train set...")
    train_embeddings = encoder.encode_batch(train_texts, batch_size=16)

    print("[generate_embeddings] Encoding test set...")
    test_embeddings = encoder.encode_batch(test_texts, batch_size=16)

    # 4) Save embeddings
    train_out = "dataset/train_embeddings.npy"
    test_out = "dataset/test_embeddings.npy"

    np.save(train_out, train_embeddings)
    np.save(test_out, test_embeddings)

    elapsed = time.time() - start_time

    # 5) Print shapes and elapsed time
    print("\n[generate_embeddings] Embedding generation complete.")
    print(f"[generate_embeddings] train_embeddings shape: {train_embeddings.shape}")
    print(f"[generate_embeddings] test_embeddings shape : {test_embeddings.shape}")
    print(f"[generate_embeddings] Saved: {train_out}")
    print(f"[generate_embeddings] Saved: {test_out}")
    print(f"[generate_embeddings] Time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
