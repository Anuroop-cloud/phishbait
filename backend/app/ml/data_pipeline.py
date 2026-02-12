"""
data_pipeline.py

Production-ready preprocessing and feature engineering pipeline
for a multilingual phishing detection system.

Handles:
  - Automatic column detection from any uploaded CSV
  - Data cleaning and normalisation
  - Handcrafted feature extraction (no embeddings)
  - Stratified train / test splitting
  - Saving processed artefacts

No GPU usage — pure CPU NumPy / scikit-learn operations.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import os
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
# 2. CLASS: DataPipeline
# ──────────────────────────────────────────────
class DataPipeline:
    """End-to-end preprocessing pipeline for multilingual phishing detection."""

    # Feature column names produced by extract_features()
    FEATURE_COLUMNS = [
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

    def __init__(self, dataset_path: str) -> None:
        """
        Parameters
        ----------
        dataset_path : str
            Path to the raw CSV dataset.
        """
        self.dataset_path = dataset_path

        # Multilingual urgency keywords (Hindi, Tamil, Telugu)
        self.urgency_words = [
            "urgent", "verify", "immediately", "act now",
            "तुरंत", "अभी", "सत्यापित",
            "உடனே", "சரிபார்க்கவும்",
            "తక్షణం", "ధృవీకరించండి",
        ]

        # Multilingual threat phrases
        self.threat_words = [
            "account suspended", "legal action", "digital arrest",
            "खाता बंद", "कानूनी कार्रवाई",
            "சட்ட நடவடிக்கை",
            "చట్టపరమైన చర్య",
        ]

        # Short-URL / suspicious domains
        self.suspicious_domains = [
            "bit.ly", "tinyurl", "rb.gy", "t.co", "goo.gl",
        ]

    # ──────────────────────────────────────────
    # 3. LOAD DATA (AUTO DETECT COLUMNS)
    # ──────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        """Load CSV, auto-detect text & label columns, clean, and return.

        Auto-detection logic
        --------------------
        * **text column** → the column whose values have the longest average
          string length (most likely free-form message text).
        * **label column** → the column with exactly 2 unique non-null values
          (binary classification target).

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with columns renamed to 'text' and 'label'.

        Raises
        ------
        FileNotFoundError
            If the dataset CSV does not exist.
        ValueError
            If auto-detection fails to find suitable columns.
        """
        # --- Check file existence ---
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.dataset_path}"
            )

        # --- Read CSV ---
        df = pd.read_csv(self.dataset_path)
        print(f"[DataPipeline] Raw columns: {list(df.columns)}")

        # --- Auto-detect text column (longest avg string length) ---
        avg_lengths = {}
        for col in df.columns:
            try:
                avg_lengths[col] = df[col].astype(str).str.len().mean()
            except Exception:
                avg_lengths[col] = 0
        text_col = max(avg_lengths, key=avg_lengths.get)

        # --- Auto-detect label column (exactly 2 unique values) ---
        label_col = None
        for col in df.columns:
            if col == text_col:
                continue
            if df[col].dropna().nunique() == 2:
                label_col = col
                break

        if label_col is None:
            raise ValueError(
                "Could not auto-detect a binary label column "
                "(expected a column with exactly 2 unique values)."
            )

        print(f"[DataPipeline] Detected text column : '{text_col}'")
        print(f"[DataPipeline] Detected label column: '{label_col}'")

        # --- Keep only the detected columns and rename ---
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]

        # --- Encode label to numeric if necessary ---
        if not pd.api.types.is_numeric_dtype(df["label"]):
            le = LabelEncoder()
            df["label"] = le.fit_transform(df["label"])
            print(f"[DataPipeline] Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # --- Basic cleaning ---
        df = df.dropna(subset=["text", "label"])    # Drop null rows
        df = df.drop_duplicates()                    # Remove duplicates
        df["text"] = df["text"].astype(str).str.strip()  # Strip whitespace & ensure str
        df = df.reset_index(drop=True)               # Reset index

        # --- Summary ---
        print(f"[DataPipeline] Total rows after cleaning: {len(df)}")
        print(f"[DataPipeline] Class distribution:\n{Counter(df['label'].tolist())}")

        return df

    # ──────────────────────────────────────────
    # 4. TEXT CLEANING
    # ──────────────────────────────────────────
    @staticmethod
    def clean_text(text: str) -> str:
        """Normalise a single text string for downstream processing.

        Steps
        -----
        1. Convert to string (safety).
        2. Lower-case the entire string.
        3. Collapse multiple whitespace into one space.
        4. Collapse repeated punctuation (e.g. '!!!!' → '!').
        5. URLs are kept intact — no removal.
        6. Strip leading / trailing whitespace.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text.
        """
        text = str(text)

        # Lowercase
        text = text.lower()

        # Collapse extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Collapse repeated punctuation (keep single occurrence)
        text = re.sub(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])\1+", r"\1", text)

        # Strip edges
        text = text.strip()

        return text

    # ──────────────────────────────────────────
    # 5. FEATURE ENGINEERING
    # ──────────────────────────────────────────
    def extract_features(self, text: str) -> np.ndarray:
        """Extract nine hand-crafted features from a single text.

        Features
        --------
        0  url_count              – number of URLs
        1  dot_count              – number of '.' characters
        2  has_at_symbol          – 1 if '@' present, else 0
        3  urgency_flag           – 1 if any urgency keyword found
        4  threat_flag            – 1 if any threat phrase found
        5  suspicious_domain_flag – 1 if any suspicious short-URL domain found
        6  caps_word_count        – fully-uppercase words (len > 1)
        7  digit_count            – total numeric characters
        8  special_char_count     – count of ! ? $ characters

        Parameters
        ----------
        text : str
            Input text (original case preserved for caps detection).

        Returns
        -------
        np.ndarray
            Feature vector of shape (9,).
        """
        text_lower = text.lower()

        # 1. URL count
        url_count = len(re.findall(r"http[s]?://\S+", text))

        # 2. Dot count
        dot_count = text.count(".")

        # 3. '@' symbol presence
        has_at_symbol = 1 if "@" in text else 0

        # 4. Urgency flag (multilingual)
        urgency_flag = 1 if any(w in text_lower for w in self.urgency_words) else 0

        # 5. Threat flag (multilingual)
        threat_flag = 1 if any(w in text_lower for w in self.threat_words) else 0

        # 6. Suspicious domain flag
        suspicious_domain_flag = (
            1 if any(d in text_lower for d in self.suspicious_domains) else 0
        )

        # 7. Fully-uppercase word count
        caps_word_count = sum(1 for w in text.split() if w.isupper() and len(w) > 1)

        # 8. Digit count
        digit_count = sum(c.isdigit() for c in text)

        # 9. Special character count (! ? $)
        special_char_count = sum(c in "!?$" for c in text)

        return np.array(
            [
                url_count,
                dot_count,
                has_at_symbol,
                urgency_flag,
                threat_flag,
                suspicious_domain_flag,
                caps_word_count,
                digit_count,
                special_char_count,
            ],
            dtype=np.float64,
        )

    # ──────────────────────────────────────────
    # 6. APPLY FEATURE ENGINEERING
    # ──────────────────────────────────────────
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning and feature extraction to the whole dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with a 'text' column.

        Returns
        -------
        pd.DataFrame
            Updated dataframe with 'clean_text' and nine feature columns.
        """
        # Clean text (lowered / normalised version)
        df["clean_text"] = df["text"].apply(self.clean_text)

        # Extract features from the *original* text (preserves case for caps detection)
        features = df["text"].apply(self.extract_features)

        # Expand the (9,) arrays into separate named columns
        feature_matrix = np.stack(features.values)
        for idx, col_name in enumerate(self.FEATURE_COLUMNS):
            df[col_name] = feature_matrix[:, idx]

        print(
            f"[DataPipeline] Feature engineering complete — "
            f"{len(self.FEATURE_COLUMNS)} features added."
        )
        return df

    # ──────────────────────────────────────────
    # 7. SPLIT DATA
    # ──────────────────────────────────────────
    @staticmethod
    def split_data(df: pd.DataFrame):
        """Stratified 80 / 20 train-test split.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'label' column.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )

        print(f"[DataPipeline] Train shape: {train_df.shape}")
        print(f"[DataPipeline] Test  shape: {test_df.shape}")

        return train_df, test_df

    # ──────────────────────────────────────────
    # 8. SAVE FILES
    # ──────────────────────────────────────────
    @staticmethod
    def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Persist the processed train and test sets as CSV files.

        Output paths
        -------------
        dataset/train_processed.csv
        dataset/test_processed.csv
        """
        output_dir = "dataset"
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train_processed.csv")
        test_path = os.path.join(output_dir, "test_processed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"[DataPipeline] Saved → {train_path}")
        print(f"[DataPipeline] Saved → {test_path}")


# ──────────────────────────────────────────────
# 9. MAIN EXECUTION BLOCK
# ──────────────────────────────────────────────
if __name__ == "__main__":

    pipeline = DataPipeline("dataset/phishing_dataset_complete (1).csv")

    df = pipeline.load_data()
    df = pipeline.apply_feature_engineering(df)
    train_df, test_df = pipeline.split_data(df)
    pipeline.save_processed(train_df, test_df)

    print("\nPreprocessing complete.")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
