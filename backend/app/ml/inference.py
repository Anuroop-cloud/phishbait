"""
inference.py

Production-ready single-text prediction for the multilingual
phishing detection system.

Dual-model architecture loaded once at module level:
  1. TEXT MODEL  — Hybrid RandomForest (IndicBERT 768-d + 9 handcrafted = 777-d)
  2. URL MODEL   — RandomForest trained on PhiUSIIL Kaggle URL features (20-d)

Both outputs feed into the explanation engine which produces a
combined verdict, threat score, confidence, and human-readable reasons.
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import json
import os
import sys
from urllib.parse import urlparse
from typing import Optional

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
from url_feature_extractor import (
    extract_urls,
    extract_url_features,
    get_triggered_features,
    URL_FEATURE_COLUMNS,
)
from explanation_engine import generate_explanation

try:
    from deep_translator import GoogleTranslator  # type: ignore
except Exception:
    GoogleTranslator = None


# ──────────────────────────────────────────────
# 2. LOAD MODELS & ENCODER ONCE (GLOBAL)
# ──────────────────────────────────────────────

# ── 2a. Text model ───────────────────────────
print("[inference] Loading hybrid RandomForest text model...")
text_model = joblib.load(os.path.join(MODEL_DIR, "hybrid_random_forest.pkl"))

print("[inference] Initializing IndicBERT encoder...")
encoder = IndicBERTEncoder()

pipeline = DataPipeline(os.path.join(DATASET_DIR, "train_processed.csv"))

# ── 2b. URL model ────────────────────────────
url_model_path = os.path.join(MODEL_DIR, "url_random_forest.pkl")
if os.path.exists(url_model_path):
    print("[inference] Loading URL RandomForest model...")
    url_model = joblib.load(url_model_path)
    print("[inference] URL model loaded OK")
else:
    url_model = None
    print("[inference] WARNING: url_random_forest.pkl not found — URL analysis disabled.")

# ── 2c. Metrics ──────────────────────────────
metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        MODEL_METRICS = json.load(f)
    print(f"[inference] Text model accuracy: {MODEL_METRICS.get('accuracy')}")
else:
    MODEL_METRICS = {"accuracy": None}
    print("[inference] WARNING: model_metrics.json not found.")

url_metrics_path = os.path.join(MODEL_DIR, "url_model_metrics.json")
if os.path.exists(url_metrics_path):
    with open(url_metrics_path, "r") as f:
        url_metrics = json.load(f)
    MODEL_METRICS["url_accuracy"] = url_metrics.get("accuracy")
    print(f"[inference] URL model accuracy: {url_metrics.get('accuracy')}")

print("Models and encoder successfully initialized.")

text_feature_cols = [
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

# ── Kaggle column order used when training the URL model ──
URL_KAGGLE_FEATURE_ORDER = [
    "URLLength", "DomainLength", "IsHTTPS", "IsDomainIP",
    "NoOfSubDomain", "TLDLength", "NoOfDegitsInURL",
    "DegitRatioInURL", "LetterRatioInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    "HasObfuscation", "NoOfLettersInURL", "ObfuscationRatio",
    "CharContinuationRate", "URLSimilarityIndex", "URLCharProb",
    "TLDLegitimateProb", "NoOfEqualsInURL", "NoOfQMarkInURL",
]

# Well-known legitimate domains that should not be over-flagged when
# there are no structural URL red flags.
TRUSTED_DOMAINS = {
    "google.com",
    "accounts.google.com",
    "github.com",
    "microsoft.com",
    "sbi.co.in",
}


LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

# Map our extracted feature names → Kaggle column positions
_OUR_TO_KAGGLE_MAP = {
    "url_length":        "URLLength",
    "domain_length":     "DomainLength",
    "is_https":          "IsHTTPS",
    "is_domain_ip":      "IsDomainIP",
    "num_subdomains":    "NoOfSubDomain",
    "tld_length":        "TLDLength",
    "num_digits_in_url": "NoOfDegitsInURL",
    "digit_ratio":       "DegitRatioInURL",
    "letter_ratio":      "LetterRatioInURL",
    "num_special_chars": "NoOfOtherSpecialCharsInURL",
    "special_char_ratio":"SpacialCharRatioInURL",
    "has_obfuscation":   "HasObfuscation",
}


def _url_features_to_kaggle_vector(features: dict) -> np.ndarray:
    """Convert our extracted URL features dict into the 20-d vector
    in the same column order the URL model was trained on.

    For columns we can't compute from a raw URL string alone
    (e.g. page-level features like LineOfCode) we use safe defaults.
    """
    vec = []
    for kaggle_col in URL_KAGGLE_FEATURE_ORDER:
        # Find the matching key from our feature dict
        found = False
        for our_key, kag_key in _OUR_TO_KAGGLE_MAP.items():
            if kag_key == kaggle_col:
                vec.append(float(features.get(our_key, 0)))
                found = True
                break
        if not found:
            # Columns we can approximate
            if kaggle_col == "NoOfLettersInURL":
                vec.append(float(sum(c.isalpha() for c in str(features.get("url_length", "")))))
                # Better: count from original URL
                url_len = features.get("url_length", 0)
                letter_r = features.get("letter_ratio", 0)
                vec[-1] = float(round(url_len * letter_r))
            elif kaggle_col == "ObfuscationRatio":
                vec.append(0.0)
            elif kaggle_col == "CharContinuationRate":
                vec.append(0.8)  # neutral default
            elif kaggle_col == "URLSimilarityIndex":
                vec.append(50.0)  # neutral default
            elif kaggle_col == "URLCharProb":
                vec.append(0.05)  # neutral default
            elif kaggle_col == "TLDLegitimateProb":
                tld_legit = features.get("tld_is_legit", 0)
                vec.append(0.5 if tld_legit else 0.01)
            elif kaggle_col == "NoOfEqualsInURL":
                vec.append(0.0)
            elif kaggle_col == "NoOfQMarkInURL":
                vec.append(0.0)
            else:
                vec.append(0.0)
    return np.array(vec, dtype=np.float64)


def _is_trusted_domain(url: str) -> bool:
    """Return True if URL belongs to a known trusted domain/subdomain."""
    try:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        host = (parsed.netloc or "").lower()
        if ":" in host:
            host = host.split(":", 1)[0]
        if host.startswith("www."):
            host = host[4:]

        return any(host == d or host.endswith(f".{d}") for d in TRUSTED_DOMAINS)
    except Exception:
        return False


def _is_url_only_input(text: str, urls: list[str]) -> bool:
    """True when user input is essentially just a URL (no extra message text)."""
    if len(urls) != 1:
        return False

    raw = (text or "").strip()
    single = urls[0].strip()

    # exact match, or exact match after normalizing trailing slash
    if raw == single:
        return True
    if raw.rstrip("/") == single.rstrip("/"):
        return True

    # if no spaces and URL is the whole input token
    if " " not in raw and raw.startswith(single):
        return True

    return False


def _detect_input_language(text: str) -> str:
    """Lightweight script-based language detection for explanation localization."""
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return "hi"
    if any("\u0B80" <= ch <= "\u0BFF" for ch in text):
        return "ta"
    if any("\u0C00" <= ch <= "\u0C7F" for ch in text):
        return "te"
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "bn"
    if any("\u0C80" <= ch <= "\u0CFF" for ch in text):
        return "kn"
    if any("\u0D00" <= ch <= "\u0D7F" for ch in text):
        return "ml"
    if any("\u0A80" <= ch <= "\u0AFF" for ch in text):
        return "gu"
    if any("\u0A00" <= ch <= "\u0A7F" for ch in text):
        return "pa"
    if any("\u0B00" <= ch <= "\u0B7F" for ch in text):
        return "or"
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "as"
    if any("\u0600" <= ch <= "\u06FF" for ch in text):
        return "ur"
    return "en"


def _translate_text(text: str, target_lang: str) -> str:
    """Translate English text to target language, with safe fallback."""
    if target_lang == "en" or not text:
        return text
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text


def _translate_list(items: list[str], target_lang: str) -> list[str]:
    return [_translate_text(i, target_lang) for i in items]


# ──────────────────────────────────────────────
# 3. PREDICTION FUNCTION
# ──────────────────────────────────────────────
def predict(text: str, language_hint: Optional[str] = None) -> dict:
    """Run dual-model phishing prediction on a single text.

    Parameters
    ----------
    text : str
        Raw input message (any supported language).

    Returns
    -------
    dict
        {
            "classification":    "Legitimate" | "Suspicious" | "Highly Likely Phishing",
            "confidence_score":  float (0.0 - 1.0),
            "reasoning":         str,
            "detected_signals":  list[str],
            "prediction":        str,
            "threat_score":      int (0-100),
            "confidence":        float (0.0 - 1.0),
            "model_accuracy":    float,
            "reasons":           list[str],
            "summary":           str,
        }
    """
    # ── A. Text model ────────────────────────
    clean = pipeline.clean_text(text)
    features = pipeline.extract_features(clean)
    embedding = encoder.encode(clean)
    text_vector = np.concatenate([embedding, features]).reshape(1, -1)

    text_pred_raw = text_model.predict(text_vector)[0]
    text_prob_all = text_model.predict_proba(text_vector)[0]
    # probability of the PHISHING class (class 1)
    text_phish_prob = float(text_prob_all[1]) if len(text_prob_all) > 1 else float(text_prob_all[0])
    text_prediction = "Phishing" if text_pred_raw == 1 else "Safe"

    # If input is just a URL token, the text model tends to overreact to
    # brand/security words. Let the URL model carry most of the signal.
    urls = extract_urls(text)
    if _is_url_only_input(text, urls):
        text_phish_prob = min(text_phish_prob, 0.20)
        text_prediction = "Safe" if text_phish_prob < 0.5 else "Phishing"

    # ── B. URL model (if URLs present) ───────
    url_probability = None
    url_prediction = None
    all_url_triggers: list[str] = []

    if urls and url_model is not None:
        url_probs = []
        trusted_url_present = False
        for u in urls:
            feats = extract_url_features(u)
            triggers = get_triggered_features(feats)
            all_url_triggers.extend(triggers)

            if _is_trusted_domain(u) and not triggers:
                trusted_url_present = True

            vec = _url_features_to_kaggle_vector(feats).reshape(1, -1)
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            prob = url_model.predict_proba(vec)[0]
            phish_p = float(prob[1]) if len(prob) > 1 else float(prob[0])

            # Dampening rule 1: if URL has no structural red flags and uses HTTPS,
            # avoid over-penalizing benign links due to model mismatch.
            if not triggers and feats.get("is_https", 0) == 1:
                phish_p = min(phish_p, 0.35)

            # Dampening rule 2: strong trust fallback for known legitimate domains
            # when no suspicious URL triggers are present.
            if not triggers and _is_trusted_domain(u):
                phish_p = min(phish_p, 0.08)

            url_probs.append(phish_p)

        # Take the maximum phishing probability across all URLs
        url_probability = max(url_probs)
        url_prediction = "Phishing" if url_probability >= 0.5 else "Safe"

        # Deduplicate triggers
        all_url_triggers = list(dict.fromkeys(all_url_triggers))
    else:
        trusted_url_present = False

    # ── C. Explanation engine ────────────────
    result = generate_explanation(
        user_text=text,
        text_probability=text_phish_prob,
        text_prediction=text_prediction,
        url_probability=url_probability,
        url_prediction=url_prediction,
        url_triggered=all_url_triggers,
        trusted_url_present=trusted_url_present,
    )

    detected_lang = (language_hint or _detect_input_language(text)).lower()
    if detected_lang not in LANGUAGE_MAP:
        detected_lang = _detect_input_language(text)
    detected_lang_label = LANGUAGE_MAP.get(detected_lang, "English")

    summary_localized = _translate_text(result.get("summary", ""), detected_lang)
    reasoning_localized = _translate_text(result.get("reasoning", ""), detected_lang)
    reasons_localized = _translate_list(result.get("reasons", []), detected_lang)

    result["detected_language"] = detected_lang
    result["detected_language_label"] = detected_lang_label
    result["summary_localized"] = summary_localized
    result["reasoning_localized"] = reasoning_localized
    result["reasons_localized"] = reasons_localized

    # Attach model accuracy
    result["model_accuracy"] = MODEL_METRICS.get("accuracy")

    return result


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        # Should be: Highly Likely Phishing
        "Your account will be suspended immediately. Click http://bit.ly/abc",
        # Should be: Legitimate
        "Meeting scheduled for tomorrow at 10 AM.",
        # Should be: Highly Likely Phishing (Hindi)
        "तुरंत अपना खाता सत्यापित करें। http://tinyurl.com/xyz",
        # Should be: Legitimate
        "Check out https://www.google.com for more info.",
        # Should be: Highly Likely Phishing
        "URGENT: Your SBI account is blocked! Verify at http://sbi-secure-login.xyz/verify?id=99",
        # Should be: Highly Likely Phishing
        "Your account has been suspended. Verify immediately at http://bank-secure-login.xyz",
        # Should be: Legitimate (informational OTP)
        "Your OTP is 482913. Do not share this OTP with anyone. Valid for 10 minutes.",
        # Should be: Legitimate (Tamil OTP)
        "உங்கள் OTP 834521. இதை யாரிடமும் பகிர வேண்டாம்.",
        # Should be: Highly Likely Phishing (OTP request)
        "Your SBI account is locked. Share your OTP to unlock: http://sbi-verify.xyz",
        # Should be: Legitimate (casual)
        "Hey, let's catch up for coffee tomorrow at 5pm?",
    ]

    for text in samples:
        result = predict(text)
        snippet = text[:70] + ("..." if len(text) > 70 else "")
        print(f"\nText: \"{snippet}\"")
        print(f"  Classification : {result['classification']}")
        print(f"  Threat Score   : {result['threat_score']}")
        print(f"  Confidence     : {result['confidence_score']}")
        print(f"  Summary        : {result['summary']}")
        print(f"  Signals        : {result['detected_signals']}")
        print(f"  Reasoning      : {result['reasoning'][:120]}...")
        print(f"  Reasons:")
        for r in result["reasons"]:
            print(f"    - {r}")

