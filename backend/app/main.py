"""FastAPI backend for multilingual phishing detection."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.ml.inference import MODEL_METRICS, predict

app = FastAPI(
    title="PhishBait — Multilingual Phishing Detector",
    version="1.0.0",
)

# Allow frontend integration (including localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    language_hint: str | None = None


@app.get("/")
def root():
    return {"status": "API running"}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """Run phishing detection on the submitted text."""
    result = predict(req.text, language_hint=req.language_hint)
    return result


@app.get("/metrics")
def get_model_metrics():
    """Return accuracy and other metrics from the last training run."""
    return MODEL_METRICS
