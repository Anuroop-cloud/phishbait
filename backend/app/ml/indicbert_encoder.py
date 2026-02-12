"""
indicbert_encoder.py

IndicBERT feature extractor for multilingual phishing detection.

Uses the pre-trained ai4bharat/indic-bert model as a frozen encoder
to produce 768-dimensional CLS embeddings. No fine-tuning is performed —
the model is used purely as a feature extractor on CPU.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────
# 2. CLASS: IndicBERTEncoder
# ──────────────────────────────────────────────
class IndicBERTEncoder:
    """Frozen IndicBERT encoder that extracts CLS-token embeddings."""

    def __init__(self) -> None:
        """Load tokenizer and model, freeze weights, move to CPU.

        Tries the requested IndicBERT first. If access is gated, it falls back
        to public multilingual models so the pipeline can still run.
        """

        self.model_name = "ai4bharat/IndicBERTv2-MLM-only"
        candidate_models = [
            "ai4bharat/IndicBERTv2-MLM-only",  # public IndicBERT v2
            "google/muril-base-cased",         # multilingual Indic fallback
        ]

        last_error = None
        self.tokenizer = None
        self.model = None

        # Load tokenizer/model from HF with graceful fallback
        for model_name in candidate_models:
            try:
                print(f"[IndicBERTEncoder] Loading tokenizer from '{model_name}' ...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                print(f"[IndicBERTEncoder] Loading model from '{model_name}' ...")
                model = AutoModel.from_pretrained(model_name)

                self.model_name = model_name
                self.tokenizer = tokenizer
                self.model = model
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[IndicBERTEncoder] Failed to load '{model_name}': {exc}")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Unable to load any encoder model.") from last_error

        # Set to evaluation mode (disables dropout, etc.)
        self.model.eval()

        # Ensure model stays on CPU
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # Disable gradient computation globally for this model
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"[IndicBERTEncoder] Model loaded and frozen on CPU: '{self.model_name}'")

    # ──────────────────────────────────────────
    # Single-text encoding
    # ──────────────────────────────────────────
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a 768-d CLS embedding.

        Parameters
        ----------
        text : str
            Input text (any supported Indic language or English).

        Returns
        -------
        np.ndarray
            1-D feature vector of shape (768,).
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (no gradient tracking)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract CLS token embedding: first token of last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Convert to 1-D numpy array
        return cls_embedding.squeeze(0).cpu().numpy()

    # ──────────────────────────────────────────
    # Batch encoding
    # ──────────────────────────────────────────
    def encode_batch(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Encode a list of texts into CLS embeddings in batches.

        Parameters
        ----------
        texts : list[str]
            List of input texts.
        batch_size : int, optional
            Number of texts per batch (default 16).

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_samples, 768).
        """
        all_embeddings: list[np.ndarray] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # CLS embeddings for the whole batch → (batch, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

            print(
                f"[IndicBERTEncoder] Encoded batch {start // batch_size + 1}"
                f" / {(total + batch_size - 1) // batch_size}"
                f"  ({end}/{total} texts)"
            )

        # Stack all batches → (n_samples, 768)
        return np.vstack(all_embeddings)


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    encoder = IndicBERTEncoder()

    # Single text
    sample = "Your account has been suspended. Verify immediately."
    vec = encoder.encode(sample)
    print(f"\nSingle encode  → shape: {vec.shape}, dtype: {vec.dtype}")

    # Batch
    samples = [
        "Your account has been suspended. Verify immediately.",
        "तुरंत अपना खाता सत्यापित करें।",
        "உங்கள் கணக்கு இடைநிறுத்தப்பட்டுள்ளது.",
        "మీ ఖాతా నిలిపివేయబడింది. వెంటనే ధృవీకరించండి.",
        "Meeting scheduled for tomorrow at 10 AM.",
    ]
    mat = encoder.encode_batch(samples)
    print(f"Batch  encode  → shape: {mat.shape}, dtype: {mat.dtype}")
