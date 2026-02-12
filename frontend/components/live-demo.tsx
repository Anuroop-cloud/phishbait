"use client"

import { useState } from "react"
import { ShieldAlert, ShieldCheck, ShieldQuestion, Loader2, Sparkles, Link2, AlertTriangle, UserCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

type Result = {
  status: "safe" | "suspicious" | "phishing"
  reasons: string[]
  language: string
  confidence: number
}

const statusConfig = {
  safe: {
    icon: ShieldCheck,
    label: "Safe Message",
    bgClass: "bg-accent/10 border-accent",
    textClass: "text-accent",
    badgeClass: "bg-accent text-accent-foreground",
  },
  suspicious: {
    icon: ShieldQuestion,
    label: "Suspicious Message",
    bgClass: "bg-secondary/10 border-secondary",
    textClass: "text-secondary",
    badgeClass: "bg-secondary text-secondary-foreground",
  },
  phishing: {
    icon: ShieldAlert,
    label: "Phishing Detected",
    bgClass: "bg-destructive/10 border-destructive",
    textClass: "text-destructive",
    badgeClass: "bg-destructive text-destructive-foreground",
  },
}

const reasonIcons: Record<string, typeof Link2> = {
  url: Link2,
  urgency: AlertTriangle,
  authority: UserCheck,
}

function analyzeMessage(text: string): Result {
  const lower = text.toLowerCase()
  const hasUrl =
    /https?:\/\/|www\.|\.com|\.in|bit\.ly/i.test(text)
  const hasUrgency =
    /तुरंत|உடனே|urgent|immediately|जल्दी|अभी|இப்போது|तत्काल|quickly|hurry/i.test(text)
  const hasAuthority =
    /rbi|sbi|bank|police|government|सरकार|पुलिस|बैंक|aadhaar|kyc|otp|verify|account.*blocked|suspended/i.test(text)
  const hasHindi = /[\u0900-\u097F]/.test(text)
  const hasTamil = /[\u0B80-\u0BFF]/.test(text)
  const hasTelugu = /[\u0C00-\u0C7F]/.test(text)

  let language = "English"
  if (hasHindi) language = "Hindi"
  else if (hasTamil) language = "Tamil"
  else if (hasTelugu) language = "Telugu"

  const reasons: string[] = []
  let score = 0

  if (hasUrl) {
    reasons.push("url:Suspicious URL detected in message")
    score += 35
  }
  if (hasUrgency) {
    reasons.push("urgency:Urgency language pattern found")
    score += 30
  }
  if (hasAuthority) {
    reasons.push("authority:Authority impersonation detected")
    score += 35
  }

  if (score === 0) {
    return { status: "safe", reasons: ["No suspicious patterns detected"], language, confidence: 92 }
  }

  return {
    status: score >= 60 ? "phishing" : "suspicious",
    reasons,
    language,
    confidence: Math.min(95, 70 + score * 0.3),
  }
}

export function LiveDemo() {
  const [message, setMessage] = useState("")
  const [result, setResult] = useState<Result | null>(null)
  const [loading, setLoading] = useState(false)

  const handleAnalyze = () => {
    if (!message.trim()) return
    setLoading(true)
    setResult(null)
    setTimeout(() => {
      setResult(analyzeMessage(message))
      setLoading(false)
    }, 1500)
  }

  const config = result ? statusConfig[result.status] : null
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

              {/* Result card */}
              {result && config && StatusIcon && (
                <div className={`mt-6 rounded-xl border-2 p-5 ${config.bgClass}`}>
                  <div className="mb-3 flex items-center gap-3">
                    <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${config.badgeClass}`}>
                      <StatusIcon className="h-5 w-5" />
                    </div>
                    <div>
                      <p className={`text-lg font-bold ${config.textClass}`}>{config.label}</p>
                      <p className="text-sm text-muted-foreground">
                        {result.language} &middot; {result.confidence}% confidence
                      </p>
                    </div>
                  </div>

                  <ul className="flex flex-col gap-2">
                    {result.reasons.map((reason, i) => {
                      const parts = reason.split(":")
                      const iconKey = parts.length > 1 ? parts[0] : null
                      const text = parts.length > 1 ? parts[1] : reason
                      const ReasonIcon = iconKey ? reasonIcons[iconKey] : null
                      return (
                        <li key={i} className="flex items-center gap-2 text-sm text-foreground">
                          {ReasonIcon ? (
                            <ReasonIcon className={`h-4 w-4 ${config.textClass}`} />
                          ) : (
                            <ShieldCheck className="h-4 w-4 text-accent" />
                          )}
                          {text}
                        </li>
                      )
                    })}
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
