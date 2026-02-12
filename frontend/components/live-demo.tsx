"use client"

import { useState } from "react"
import {
  ShieldAlert,
  ShieldCheck,
  Loader2,
  Sparkles,
  AlertTriangle,
  CircleCheck,
  BrainCircuit,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { analyzeText, type PredictResponse } from "@/lib/api"

const statusConfig = {
  "Highly Likely Phishing": {
    icon: ShieldAlert,
    label: "Phishing Detected",
    bgClass: "bg-destructive/10 border-destructive",
    textClass: "text-destructive",
    badgeClass: "bg-destructive text-destructive-foreground",
  },
  "Suspicious": {
    icon: AlertTriangle,
    label: "Suspicious Message",
    bgClass: "bg-yellow-500/10 border-yellow-500",
    textClass: "text-yellow-600 dark:text-yellow-400",
    badgeClass: "bg-yellow-500 text-white",
  },
  "Legitimate": {
    icon: ShieldCheck,
    label: "Legitimate Message",
    bgClass: "bg-accent/10 border-accent",
    textClass: "text-accent",
    badgeClass: "bg-accent text-accent-foreground",
  },
}

export function LiveDemo() {
  const [message, setMessage] = useState("")
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    if (!message.trim()) return
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const data = await analyzeText(message)
      setResult(data)
    } catch {
      setError("Something went wrong. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const config = result ? statusConfig[result.classification] : null
  const StatusIcon = config?.icon

  return (
    <section id="demo" className="mandala-bg-subtle py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 lg:px-8">
        <div className="mx-auto mb-14 max-w-2xl text-center">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wider text-primary">
            Live Demo
          </p>
          <h2 className="mb-4 text-balance text-3xl font-bold text-foreground md:text-4xl">
            Test a Suspicious Message
          </h2>
          <p className="text-pretty text-lg text-muted-foreground">
            Paste any message you received and our AI will analyze it for phishing patterns in multiple Indian
            languages.
          </p>
        </div>

        <div className="mx-auto max-w-2xl">
          <Card className="overflow-hidden rounded-2xl border border-border shadow-lg">
            <CardContent className="p-6">
              {/* Input area */}
              <div className="relative">
                <textarea
                  value={message}
                  onChange={(e) => {
                    setMessage(e.target.value)
                    setResult(null)
                  }}
                  rows={5}
                  className="w-full resize-none rounded-xl border border-border bg-muted p-4 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                  placeholder="Paste a suspicious message here... (try Hindi, Tamil, or English)"
                />
                {/* Language auto-detect */}
                {message.trim() && (
                  <div className="absolute bottom-3 right-3 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                    <Sparkles className="mr-1 inline h-3 w-3" />
                    {/[\u0900-\u097F]/.test(message)
                      ? "Hindi detected"
                      : /[\u0B80-\u0BFF]/.test(message)
                        ? "Tamil detected"
                        : /[\u0C00-\u0C7F]/.test(message)
                          ? "Telugu detected"
                          : "English detected"}
                  </div>
                )}
              </div>

              {/* Analyze button */}
              <Button
                onClick={handleAnalyze}
                disabled={!message.trim() || loading}
                className="mt-4 w-full bg-primary text-primary-foreground hover:bg-primary/85"
                size="lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Analyze Message"
                )}
              </Button>

              {/* Error */}
              {error && (
                <div className="mt-6 rounded-xl border-2 border-destructive/40 bg-destructive/5 p-5">
                  <div className="flex items-center gap-2 text-sm text-destructive">
                    <AlertTriangle className="h-4 w-4" />
                    {error}
                  </div>
                </div>
              )}

              {/* Result card */}
              {result && config && StatusIcon && (
                <div className={`mt-6 rounded-xl border-2 p-5 ${config.bgClass}`}>
                  {/* Header row: status + threat score */}
                  <div className="mb-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${config.badgeClass}`}>
                        <StatusIcon className="h-5 w-5" />
                      </div>
                      <div>
                        <p className={`text-lg font-bold ${config.textClass}`}>{config.label}</p>
                        <p className="text-sm text-muted-foreground">
                          {(result.confidence * 100).toFixed(1)}% confidence
                        </p>
                      </div>
                    </div>
                    {/* Threat Score Circle */}
                    <div className="flex flex-col items-center">
                      <div className={`flex h-14 w-14 items-center justify-center rounded-full ${config.badgeClass}`}>
                        <span className="text-xl font-extrabold">{result.threat_score}</span>
                      </div>
                      <span className="mt-1 text-xs text-muted-foreground">Threat Score</span>
                    </div>
                  </div>

                  {/* Summary */}
                  {result.summary && (
                    <p className="mb-4 text-sm font-medium text-foreground/80">{result.summary}</p>
                  )}

                  {/* Reasoning */}
                  {result.reasoning && (
                    <p className="mb-4 text-sm italic text-muted-foreground leading-relaxed">{result.reasoning}</p>
                  )}

                  {/* Detected signals */}
                  {result.detected_signals && result.detected_signals.length > 0 && (
                    <div className="mb-4 flex flex-wrap gap-1.5">
                      {result.detected_signals.map((signal, i) => (
                        <span key={i} className="rounded-full border border-border bg-muted/60 px-2.5 py-0.5 text-xs text-muted-foreground">
                          {signal.replace(/_/g, " ")}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Model accuracy badge */}
                  {result.model_accuracy != null && (
                    <div className="mb-4 inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/60 px-3 py-1 text-xs text-muted-foreground">
                      <BrainCircuit className="h-3.5 w-3.5" />
                      Model Accuracy: {(result.model_accuracy * 100).toFixed(1)}%
                    </div>
                  )}

                  {/* Reasons */}
                  <ul className="flex flex-col gap-2">
                    {result.reasons.map((reason, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-foreground">
                        {result.classification === "Highly Likely Phishing" ? (
                          <AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${config.textClass}`} />
                        ) : result.classification === "Suspicious" ? (
                          <AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${config.textClass}`} />
                        ) : (
                          <CircleCheck className={`mt-0.5 h-4 w-4 shrink-0 ${config.textClass}`} />
                        )}
                        {reason}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Sample messages */}
              <div className="mt-5 border-t border-border pt-4">
                <p className="mb-3 text-xs font-medium text-muted-foreground">Try a sample message:</p>
                <div className="flex flex-wrap gap-2">
                  {[
                    "आपका बैंक अकाउंट ब्लॉक हो गया है! तुरंत KYC अपडेट करें: bit.ly/verify-sbi",
                    "உங்கள் OTP 834521. இதை யாரிடமும் பகிர வேண்டாம்.",
                    "Hey, let's catch up for coffee tomorrow at 5pm?",
                  ].map((sample) => (
                    <button
                      key={sample}
                      onClick={() => {
                        setMessage(sample)
                        setResult(null)
                      }}
                      className="rounded-lg border border-border bg-muted px-3 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-muted/80"
                    >
                      {sample.length > 50 ? sample.slice(0, 50) + "..." : sample}
                    </button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
