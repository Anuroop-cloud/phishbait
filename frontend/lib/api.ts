const BASE_URL = "http://127.0.0.1:8000";

function detectLanguageCode(text: string): string {
  if (/[\u0900-\u097F]/.test(text)) return "hi";
  if (/[\u0B80-\u0BFF]/.test(text)) return "ta";
  if (/[\u0C00-\u0C7F]/.test(text)) return "te";
  if (/[\u0980-\u09FF]/.test(text)) return "bn";
  if (/[\u0C80-\u0CFF]/.test(text)) return "kn";
  if (/[\u0D00-\u0D7F]/.test(text)) return "ml";
  if (/[\u0A80-\u0AFF]/.test(text)) return "gu";
  if (/[\u0A00-\u0A7F]/.test(text)) return "pa";
  if (/[\u0600-\u06FF]/.test(text)) return "ur";
  return "en";
}

// ── Types ────────────────────────────────────────────────
export interface PredictResponse {
  classification: "Legitimate" | "Suspicious" | "Highly Likely Phishing";
  confidence_score: number;    // 0 - 1
  reasoning: string;
  detected_signals: string[];
  prediction: string;
  threat_score: number;        // 0 - 100
  confidence: number;          // 0 - 1
  model_accuracy: number;      // 0 - 1
  reasons: string[];
  summary: string;
  detected_language?: string;
  detected_language_label?: string;
  summary_localized?: string;
  reasoning_localized?: string;
  reasons_localized?: string[];
}

export interface MetricsResponse {
  accuracy: number;
  feature_dimension: number;
  train_samples: number;
  test_samples: number;
  url_accuracy?: number;
}

// ── API functions ────────────────────────────────────────
export async function analyzeText(text: string): Promise<PredictResponse> {
  const language_hint = detectLanguageCode(text);
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language_hint }),
  });

  if (!response.ok) {
    throw new Error("Prediction failed");
  }

  const data = await response.json();

  return {
    classification: data.classification,
    confidence_score: data.confidence_score,
    reasoning: data.reasoning,
    detected_signals: data.detected_signals,
    prediction: data.prediction,
    confidence: data.confidence,
    model_accuracy: data.model_accuracy,
    threat_score: data.threat_score,
    reasons: data.reasons,
    summary: data.summary,
    detected_language: data.detected_language,
    detected_language_label: data.detected_language_label,
    summary_localized: data.summary_localized,
    reasoning_localized: data.reasoning_localized,
    reasons_localized: data.reasons_localized,
  };
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${BASE_URL}/metrics`);

  if (!response.ok) {
    throw new Error("Metrics fetch failed");
  }

  return await response.json();
}
