# phishbait

# SurakshaAI

A multilingual phishing detection platform built in a 36-hour hackathon to detect scam messages in Indian language contexts and provide explainable, user-friendly risk outputs.

---

## Overview

SurakshaAI combines a modern frontend, API backend, and hybrid ML pipeline to classify messages into:

- **Legitimate**
- **Suspicious**
- **Highly Likely Phishing**

It is designed for Indian multilingual usage (Hindi, Tamil, Telugu, Urdu, etc.), with explainable outputs and localized reasoning.

---

## Problem Statement

Most phishing detectors are English-centric and often miss code-mixed or regional-language scam patterns.  
SurakshaAI addresses this by combining multilingual text understanding, phishing-focused handcrafted features, URL risk intelligence, and explanation-first outputs.

---

## Key Features

- Real-time phishing message analysis
- Multilingual support for Indian language contexts
- Hybrid detection logic (text + URL signals)
- Confidence score (0–1) and threat score (0–100)
- Explainable reasoning with machine-readable signals
- Native-language explanation + English toggle
- REST API integration with frontend

---

## System Architecture (High-Level)

1. User enters message in frontend
2. Frontend sends `text` + `language_hint` to FastAPI backend
3. Backend runs:
   - text preprocessing + feature extraction
   - IndicBERT embedding extraction
   - text model scoring
   - URL extraction + URL model scoring
4. Explanation engine fuses all signals into final verdict
5. API returns classification, confidence, threat score, reasoning, localized explanation fields

```text
UI (Next.js) -> FastAPI -> ML Inference
                         |- Text model (IndicBERT + handcrafted features)
                         |- URL model (structural URL features)
                         |- Explanation engine (fusion + reasoning + scoring)
                      -> JSON response -> UI rendering
```

---

## ML Workflow

### 1) Data Pipeline
- Auto-detect text/label columns from raw CSV
- Clean text/labels
- Remove nulls and duplicates
- Encode labels
- Stratified train/test split (80/20)

### 2) Feature Engineering
- 9 handcrafted phishing features from message text:
  - `url_count`
  - `dot_count`
  - `has_at_symbol`
  - `urgency_flag`
  - `threat_flag`
  - `suspicious_domain_flag`
  - `caps_word_count`
  - `digit_count`
  - `special_char_count`

### 3) Multilingual Text Representation
- Frozen **IndicBERT** encoder for 768-d CLS embeddings
- No fine-tuning in current pipeline (feature extractor mode)

### 4) Hybrid Text Model Training
- Concatenate:
  - 768-d IndicBERT embeddings
  - 9 handcrafted features
- Final vector size: **777**
- Classifier: RandomForest (`n_estimators=200`, `class_weight="balanced"`)

### 5) URL Model Training
- Separate RandomForest model trained on 20 structural URL features
- Used during inference when URLs are present

### 6) Inference Fusion
- Combines text model probability + URL model probability + rule-based risk signals
- Produces:
  - final class
  - confidence
  - threat score
  - explanation reasons and summary

---

## Feature Engineering Strategy

SurakshaAI uses practical phishing indicators beyond generic NLP:

- URL presence and suspicious-domain patterns
- Urgency/threat language cues
- Authority impersonation context
- Sensitive info request detection (OTP/password/CVV/PIN patterns)
- Character-level anomaly signals (caps, digits, punctuation)
- OTP context disambiguation:
  - informational OTP notification vs malicious OTP request

---

## Hybrid Inference Logic

Inference is dual-model + rules:

- **Text model:** phishing probability from 777-d representation
- **URL model:** phishing probability from structural URL features
- **Rule layer:** signal detection and guardrails for better real-world behavior

Output classes:
- `Legitimate`
- `Suspicious`
- `Highly Likely Phishing`

With:
- `confidence_score` (0–1)
- `threat_score` (0–100)
- `reasoning`, `reasons`, and `detected_signals`

---

## Multilingual Explanation System

The backend supports localization of explanation output:

- Detects input script or accepts frontend `language_hint`
- Returns:
  - `summary_localized`
  - `reasoning_localized`
  - `reasons_localized`
- Frontend can toggle localized explanation <-> English

---

## API Endpoints

### `GET /`
Health check.

**Response**
```json
{
  "status": "API running"
}
```

### `POST /predict`
Run phishing prediction and explanation.

**Request**
```json
{
  "text": "आपका बैंक अकाउंट ब्लॉक हो गया है! तुरंत KYC अपडेट करें: bit.ly/verify-sbi",
  "language_hint": "hi"
}
```

**Response (example)**
```json
{
  "classification": "Highly Likely Phishing",
  "confidence_score": 0.91,
  "reasoning": "Multiple phishing indicators were detected in this message...",
  "detected_signals": ["urgency", "external_link", "impersonation"],
  "prediction": "Highly Likely Phishing",
  "threat_score": 91,
  "confidence": 0.91,
  "reasons": [
    "Urgency language detected.",
    "URL risk: URL uses a known suspicious or free-hosting domain.",
    "Analysed using IndicBERT multilingual text model and URL structural analysis model."
  ],
  "summary": "This message is highly likely a phishing or scam attempt.",
  "detected_language": "hi",
  "detected_language_label": "Hindi",
  "summary_localized": "यह संदेश फ़िशिंग या स्कैम होने की बहुत अधिक संभावना रखता है।",
  "reasoning_localized": "इस संदेश में कई फ़िशिंग संकेत मिले हैं...",
  "reasons_localized": [
    "तत्कालता की भाषा पाई गई।",
    "URL जोखिम संकेत मिले।"
  ],
  "model_accuracy": 0.995556
}
```

### `GET /metrics`
Returns saved model metrics.

**Response (example)**
```json
{
  "accuracy": 0.995556,
  "precision": 0.995556,
  "recall": 0.995556,
  "f1_score": 0.995556,
  "feature_dimension": 777,
  "train_samples": 3597,
  "test_samples": 900,
  "url_accuracy": 0.9998515659789223
}
```

---

## Tech Stack

### Frontend
- Next.js
- TypeScript
- Tailwind CSS
- React

### Backend
- FastAPI
- Pydantic
- Uvicorn

### ML/Data
- Python
- NumPy
- Pandas
- scikit-learn
- joblib
- transformers
- torch
- deep-translator

---

## Installation & Setup

## Prerequisites
- Python 3.10+
- Node.js 18+
- pnpm (or npm)

## 1. Clone
```bash
git clone <repo-url>
cd phishbait
```

## 2. Backend setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
pip install fastapi uvicorn pydantic numpy pandas scikit-learn joblib torch transformers
```

## 3. Frontend setup
```bash
cd frontend
pnpm install
# or npm install
```

---

## How to Run Locally

## Start backend (from project root)
```bash
.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000
```

## Start frontend (from frontend directory)
```bash
pnpm dev
# or npm run dev
```

- Frontend: http://localhost:3000  
- Backend: http://127.0.0.1:8000

---

## Training Pipeline Commands (Optional)

From project root:

```bash
python backend/app/ml/data_pipeline.py
python backend/app/ml/generate_embeddings.py
python backend/app/ml/train_hybrid_model.py
python backend/app/ml/train_url_model.py
```

Alternative end-to-end script:
```bash
python backend/app/ml/train_augmented.py
```

---

## Evaluation Metrics (Current Saved)

### Hybrid Text Model
- Accuracy: **0.995556**
- Precision: **0.995556**
- Recall: **0.995556**
- F1 Score: **0.995556**
- Feature dimension: **777**
- Train/Test: **3597 / 900**

### URL Model
- Accuracy: **0.9998515659789223**
- Feature count: **20**
- Train/Test: **188636 / 47159**

---

## Future Improvements

- Expand multilingual and code-mixed scam datasets
- Better calibration and threshold tuning across classes
- Stronger transliteration and dialect handling
- Add CI checks for model/data drift
- Containerized deployment (Docker)
- Add auth/rate limiting for production API
- Add benchmark suite across Indian language categories

---

## Team & Contributions

Built with:
- **Neerav**
- **Naman**
- **Hishaam**

ML Engineering led by **Anuroop Phukan**:
- Designed and implemented ML pipeline
- Built feature engineering logic (URL flags, suspicious domains, urgency cues, character patterns, multilingual keywords)
- Developed hybrid inference and threat scoring logic
- Implemented explainable reasoning signals
- Exposed inference via FastAPI endpoints
- Integrated ML outputs with frontend UX

---

## License

No license file is currently included.  
If open-sourcing, add a `LICENSE` file (for example MIT or Apache-2.0) and update this section.