const BASE_URL = "http://127.0.0.1:8000";

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
  const response = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
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
  };
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${BASE_URL}/metrics`);

  if (!response.ok) {
    throw new Error("Metrics fetch failed");
  }

  return await response.json();
}
