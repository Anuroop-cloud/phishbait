"""
explanation_engine.py

Cybersecurity AI Explanation Assistant.

Generates clear, professional, context-aware explanations for phishing
detection results.  Combines evidence from two models:

  1. **Text model** -- IndicBERT + handcrafted features on message content
  2. **URL model** -- structural analysis of any URLs found in the message

Classification
--------------
* "Highly Likely Phishing"  -- combined score >= 0.7
* "Suspicious"              -- combined score 0.4 - 0.7
* "Legitimate"              -- combined score < 0.4

Important context rules
-----------------------
* Do NOT classify based solely on keyword presence ("OTP", "bank", etc.).
* Distinguish between informational OTP notifications (legitimate) and
  requests to share/enter OTP via link (phishing).
* "Do not share this OTP" => strong legitimacy signal.
* Legitimate companies never ask users to share OTP via SMS/external links.
* If unsure, classify as Suspicious instead of Highly Likely Phishing.
"""

from __future__ import annotations
import re
from typing import Optional


# ------------------------------------------------------------------ #
# Multilingual keyword / phrase lists
# ------------------------------------------------------------------ #
URGENCY_KEYWORDS = [
    "urgent", "immediately", "act now", "hurry",
    "quickly", "limited time", "expire",
    # Hindi
    "तुरंत", "अभी", "जल्दी",
    # Tamil
    "உடனே",
    # Telugu
    "తక్షణం",
]

THREAT_PHRASES = [
    "account suspended", "legal action", "digital arrest",
    "account blocked", "will be closed", "permanently deleted",
    "account will be", "action will be taken",
    # Hindi
    "खाता बंद", "कानूनी कार्रवाई",
    # Tamil
    "சட்ட நடவடிக்கை",
    # Telugu
    "చట్టపరమైన చర్య",
]

AUTHORITY_KEYWORDS = [
    "rbi", "sbi", "police", "government", "aadhaar",
    "kyc", "pan card", "income tax",
    # Hindi
    "सरकार", "पुलिस",
]

MONEY_KEYWORDS = [
    "prize", "lottery", "won", "reward", "cashback",
    "lakh", "crore",
    "credit card", "debit card",
]

# Patterns that REQUEST the user to share sensitive info
SENSITIVE_REQUEST_PATTERNS = [
    r"share\s+(your\s+)?otp",
    r"enter\s+(your\s+)?otp",
    r"send\s+(your\s+)?otp",
    r"tell\s+(us\s+)?(your\s+)?otp",
    r"provide\s+(your\s+)?otp",
    r"share\s+(your\s+)?password",
    r"enter\s+(your\s+)?password",
    r"share\s+(your\s+)?cvv",
    r"enter\s+(your\s+)?cvv",
    r"share\s+(your\s+)?pin",
    r"enter\s+(your\s+)?card\s+number",
    # Hindi
    r"otp\s+भेजें",
    r"otp\s+बताएं",
    r"otp\s+दें",
    r"पासवर्ड\s+भेजें",
    r"ओटीपी\s+भेजें",
    r"ओटीपी\s+बताएं",
]

# Patterns that indicate an informational OTP notification (LEGITIMATE)
OTP_LEGIT_PATTERNS = [
    r"do\s+not\s+share",
    r"don'?t\s+share",
    r"never\s+share",
    r"please\s+do\s+not\s+share",
    r"valid\s+for\s+\d+\s+min",
    r"expires?\s+in\s+\d+",
    r"one.?time\s+password",
    r"verification\s+code",
    r"otp\s+is\s+\d{4,8}",
    r"your\s+otp\s+is",
    r"otp\s*:\s*\d{4,8}",
    # Tamil
    r"யாரிடமும்\s+பகிர\s+வேண்டாம்",
    r"பகிர\s+வேண்டாம்",
    # Hindi
    r"किसी\s+को\s+न\s+बताएं",
    r"साझा\s+न\s+करें",
    r"शेयर\s+न\s+करें",
    r"शेयर\s+मत\s+करें",
]

EMOTIONAL_MANIPULATION = [
    "congratulations", "you have been selected", "lucky winner",
    "claim now", "claim your", "selected for",
    "बधाई हो", "आप चुने गए",
]


# ------------------------------------------------------------------ #
# Signal detection helpers
# ------------------------------------------------------------------ #
def _detect_signals(text: str, has_urls: bool) -> tuple[list[str], list[str]]:
    """Return (detected_signals: list[str], human_reasons: list[str]).

    detected_signals uses short machine-readable tags.
    human_reasons are user-facing explanations.
    """
    lower = text.lower()
    signals: list[str] = []
    reasons: list[str] = []

    # 1. Urgency
    found_urgency = [w for w in URGENCY_KEYWORDS if w in lower]
    if found_urgency:
        signals.append("urgency")
        reasons.append(
            f'Urgency language detected: uses words like "{found_urgency[0]}".'
        )

    # 2. Threats
    found_threats = [w for w in THREAT_PHRASES if w in lower]
    if found_threats:
        signals.append("threat")
        reasons.append(
            f'Threatening language detected: mentions "{found_threats[0]}".'
        )

    # 3. OTP / sensitive info -- context-aware
    otp_mentioned = bool(re.search(r"\botp\b|ओटीपी", lower))
    is_otp_request = any(re.search(p, lower) for p in SENSITIVE_REQUEST_PATTERNS)
    is_otp_legit = any(re.search(p, lower) for p in OTP_LEGIT_PATTERNS)

    if is_otp_request:
        signals.append("otp_request")
        reasons.append(
            "Requests user to share/enter OTP -- legitimate services never ask this."
        )
    elif otp_mentioned and is_otp_legit:
        signals.append("otp_informational")
        reasons.append(
            'OTP notification with "do not share" advisory -- consistent with legitimate services.'
        )
    # If OTP is just mentioned without request or legit pattern, don't flag

    # Non-OTP sensitive info requests (password, cvv, pin, card)
    sensitive_kw = ["password", "cvv", "pin", "card"]
    password_patterns = [
        p for p in SENSITIVE_REQUEST_PATTERNS
        if any(k in p for k in sensitive_kw)
    ]
    password_req = any(re.search(p, lower) for p in password_patterns)
    if password_req:
        if "sensitive_info_request" not in signals:
            signals.append("sensitive_info_request")
            reasons.append(
                "Requests sensitive personal information (password, CVV, PIN, or card number)."
            )

    # 4. Authority impersonation -- only flag when combined with other red flags
    found_auth = [w for w in AUTHORITY_KEYWORDS if w in lower]
    if found_auth:
        has_red_flags = any(
            s in signals
            for s in ["urgency", "threat", "otp_request", "sensitive_info_request"]
        )
        if has_red_flags or has_urls:
            signals.append("impersonation")
            reasons.append(
                f'References authority/institution "{found_auth[0]}" combined with other red flags.'
            )

    # 5. External link presence
    if has_urls:
        signals.append("external_link")
        # URL-specific reasons are added via url_triggered, not here

    # 6. Money / reward bait
    found_money = [w for w in MONEY_KEYWORDS if w in lower]
    if found_money:
        signals.append("financial_bait")
        reasons.append(
            f'Financial bait detected: mentions "{found_money[0]}".'
        )

    # 7. Emotional manipulation
    found_emotion = [w for w in EMOTIONAL_MANIPULATION if w in lower]
    if found_emotion:
        signals.append("emotional_manipulation")
        reasons.append(
            f'Emotional manipulation: "{found_emotion[0]}".'
        )

    # 8. Excessive CAPS
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(caps_words) >= 3:
        signals.append("excessive_caps")
        reasons.append(
            f"Excessive capitalisation ({len(caps_words)} ALL-CAPS words) -- creates false urgency."
        )

    # 9. Poor grammar heuristic
    double_space = "  " in text
    mixed_case_mid = bool(re.search(r"[a-z][A-Z]{2,}[a-z]", text))
    if double_space and mixed_case_mid:
        signals.append("poor_grammar")
        reasons.append("Unusual formatting/grammar patterns detected.")

    return signals, reasons


# ------------------------------------------------------------------ #
# Main explanation generator
# ------------------------------------------------------------------ #
def generate_explanation(
    *,
    user_text: str,
    text_probability: float,
    text_prediction: str,           # "Phishing" | "Safe"
    url_probability: float | None,
    url_prediction: str | None,     # "Phishing" | "Safe" | None
    url_triggered: list[str],       # from get_triggered_features()
    text_features: dict | None = None,
) -> dict:
    """Build a structured explanation from both model outputs.

    Returns
    -------
    dict
        classification   : "Legitimate" | "Suspicious" | "Highly Likely Phishing"
        confidence_score : float 0-1
        reasoning        : str  (paragraph explaining the verdict)
        detected_signals : list[str]  (machine-readable tags)
        threat_score     : int 0-100
        reasons          : list[str]  (bullet-point explanations)
        summary          : str  (one-line verdict)
        prediction       : str  (mapped from classification for frontend)
    """
    has_url_model = url_probability is not None
    has_urls = has_url_model or bool(url_triggered)

    # ---- Detect signals ----
    detected_signals, text_reasons = _detect_signals(user_text, has_urls)

    # ---- Context-aware legitimacy check ----
    is_likely_legit_otp = "otp_informational" in detected_signals

    # ---- Combine model probabilities ----
    # Only count red-flag signals (not informational/neutral ones)
    red_flag_signals = [
        s for s in detected_signals
        if s not in ("otp_informational", "external_link")
    ]
    rule_boost = min(0.15, len(red_flag_signals) * 0.05)

    # Legitimacy dampener for genuine OTP messages
    legit_dampener = 0.35 if is_likely_legit_otp else 0.0

    if has_url_model:
        base = max(text_probability, url_probability)

        agreement_bonus = 0.0
        if text_probability >= 0.5 and url_probability >= 0.5:
            agreement_bonus = 0.10
        elif text_probability >= 0.5 and url_triggered:
            agreement_bonus = 0.08

        combined_prob = base + agreement_bonus + rule_boost - legit_dampener
    else:
        combined_prob = text_probability + rule_boost - legit_dampener

    combined_prob = max(0.0, min(1.0, combined_prob))

    # ---- Determine classification ----
    no_red_flags = not red_flag_signals and not url_triggered

    if is_likely_legit_otp and combined_prob < 0.6:
        classification = "Legitimate"
        summary = "This appears to be a legitimate OTP/notification message."
    elif no_red_flags and combined_prob < 0.6:
        classification = "Legitimate"
        summary = "No strong phishing indicators were detected in this message."
    elif combined_prob >= 0.7:
        classification = "Highly Likely Phishing"
        summary = "This message is highly likely a phishing or scam attempt."
    elif combined_prob >= 0.4:
        classification = "Suspicious"
        summary = "This message contains suspicious elements -- exercise caution."
    else:
        classification = "Legitimate"
        summary = "No strong phishing indicators were detected in this message."

    threat_score = min(100, max(0, round(combined_prob * 100)))
    if classification == "Legitimate" and threat_score > 35:
        threat_score = max(10, threat_score - 25)

    # ---- Build reasons list ----
    reasons: list[str] = list(text_reasons)

    # Text model confidence
    if text_probability >= 0.7:
        reasons.append(
            f"Text analysis model flagged the message content "
            f"(confidence: {text_probability*100:.0f}%)."
        )
    elif text_probability >= 0.4:
        reasons.append(
            f"Text analysis detected some patterns "
            f"(confidence: {text_probability*100:.0f}%)."
        )

    # URL model reasons
    if has_url_model and url_triggered:
        for trigger in url_triggered:
            reasons.append(f"URL risk: {trigger}")

    if has_url_model and url_probability is not None:
        if url_probability >= 0.7:
            reasons.append(
                f"URL structure analysis classified the link as phishing "
                f"(confidence: {url_probability*100:.0f}%)."
            )
        elif url_probability >= 0.4:
            reasons.append(
                f"URL shows some structural anomalies "
                f"(confidence: {url_probability*100:.0f}%)."
            )

    # Positive reasons for legitimate messages
    if classification == "Legitimate" and not reasons:
        reasons.append("No phishing patterns detected in the message text.")
        if not has_url_model:
            reasons.append("No URLs were found in the message.")
        else:
            reasons.append("URLs in the message appear to be legitimate.")

    # Analysis note
    analysis_note = "Analysed using IndicBERT multilingual text model"
    if has_url_model:
        analysis_note += " and URL structural analysis model"
    analysis_note += "."
    reasons.append(analysis_note)

    # ---- Build reasoning paragraph ----
    reasoning = _build_reasoning(
        classification, combined_prob, detected_signals,
        text_probability, url_probability, is_likely_legit_otp,
        has_urls,
    )

    # ---- Map classification to frontend prediction key ----
    prediction = classification  # "Legitimate" | "Suspicious" | "Highly Likely Phishing"

    return {
        # Strict JSON output fields
        "classification": classification,
        "confidence_score": round(combined_prob, 4),
        "reasoning": reasoning,
        "detected_signals": detected_signals,
        # Display / compatibility fields
        "prediction": prediction,
        "threat_score": threat_score,
        "confidence": round(combined_prob, 4),
        "reasons": reasons,
        "summary": summary,
    }


def _build_reasoning(
    classification: str,
    combined_prob: float,
    signals: list[str],
    text_prob: float,
    url_prob: float | None,
    is_legit_otp: bool,
    has_urls: bool,
) -> str:
    """Generate a human-readable reasoning paragraph."""
    parts: list[str] = []

    if classification == "Legitimate":
        if is_legit_otp:
            parts.append(
                "This message appears to be a standard OTP or verification "
                "notification. It contains a 'do not share' advisory, which "
                "is consistent with legitimate banking and service notifications."
            )
        else:
            parts.append(
                "The message does not exhibit strong phishing characteristics."
            )
        if not signals or signals == ["otp_informational"]:
            parts.append(
                "No urgency language, threats, impersonation attempts, or "
                "suspicious links were detected."
            )

    elif classification == "Suspicious":
        parts.append(
            "The message contains some elements that could indicate phishing, "
            "but there is not enough evidence for a definitive classification."
        )
        if "urgency" in signals:
            parts.append("Urgency language was detected, which is common in scam messages.")
        if "external_link" in signals:
            parts.append("The message contains external links that should be verified before clicking.")
        parts.append("Exercise caution and verify the sender through official channels.")

    else:  # Highly Likely Phishing
        parts.append(
            "Multiple phishing indicators were detected in this message."
        )
        if "otp_request" in signals:
            parts.append(
                "The message explicitly asks the user to share or enter an OTP, "
                "which legitimate services never do via SMS or external links."
            )
        if "threat" in signals:
            parts.append("Threatening language is used to create panic and force immediate action.")
        if "impersonation" in signals:
            parts.append("The message impersonates a known authority or institution.")
        if "external_link" in signals:
            parts.append("Suspicious external links are present that could lead to credential theft.")
        if "financial_bait" in signals:
            parts.append("Financial rewards are used as bait to lure the victim.")

    # Model confidence note
    model_note = f"Text model confidence: {text_prob*100:.0f}%"
    if url_prob is not None:
        model_note += f", URL model confidence: {url_prob*100:.0f}%"
    model_note += f". Combined score: {combined_prob*100:.0f}%."
    parts.append(model_note)

    return " ".join(parts)
