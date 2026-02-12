const BASE_URL = "http://127.0.0.1:8000";

// ── Types ────────────────────────────────────────────────
export interface PredictResponse {
  prediction: "Phishing" | "Safe";
  threat_score: number;       // 0 – 100
  confidence: number;         // 0 – 1
  model_accuracy: number;     // 0 – 1
  reasons: string[];
}

export interface MetricsResponse {
  accuracy: number;
  feature_dimension: number;
  train_samples: number;
  test_samples: number;
}

// ── Helpers (derive fields the backend doesn't return) ───
function deriveThreatScore(prediction: string, confidence: number): number {
  if (prediction === "Phishing") {
    return Math.round(confidence * 100);
  }
  return Math.round((1 - confidence) * 100);
}

function deriveReasons(prediction: string, confidence: number): string[] {
  if (prediction === "Safe") {
    return ["No phishing patterns detected in this message."];
  }

  const reasons: string[] = [];
  const pct = Math.round(confidence * 100);

  reasons.push(`Model classified this message as phishing with ${pct}% confidence.`);

  if (confidence >= 0.9) {
    reasons.push("Very high probability of being a scam or phishing attempt.");
  } else if (confidence >= 0.75) {
    reasons.push("Strong phishing signals detected by the AI model.");
  } else {
    reasons.push("Some phishing indicators found — exercise caution.");
  }

  reasons.push(
    "Analysis used IndicBERT multilingual embeddings combined with handcrafted threat features."
  );

  return reasons;
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

  // Enrich the response with derived fields
  return {
    prediction: data.prediction,
    confidence: data.confidence,
    model_accuracy: data.model_accuracy,
    threat_score: deriveThreatScore(data.prediction, data.confidence),
    reasons: deriveReasons(data.prediction, data.confidence),
  };
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${BASE_URL}/metrics`);

  if (!response.ok) {
    throw new Error("Metrics fetch failed");
  }

  return await response.json();
}
