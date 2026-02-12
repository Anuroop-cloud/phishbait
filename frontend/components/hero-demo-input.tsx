"use client"

import { useState, useEffect, useCallback } from "react"
import {
  ShieldAlert,
  ShieldCheck,
  Loader2,
  Sparkles,
  AlertTriangle,
  CircleCheck,
  Gauge,
  BrainCircuit,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { analyzeText, type PredictResponse } from "@/lib/api"

const statusConfig = {
  "Highly Likely Phishing": {
    icon: ShieldAlert,
    label: "Phishing Detected",
    bgClass: "bg-destructive/10 border-destructive",
    textClass: "text-destructive",
    badgeClass: "bg-destructive text-destructive-foreground",
    scoreBg: "bg-destructive",
  },
  "Suspicious": {
    icon: AlertTriangle,
    label: "Suspicious Message",
    bgClass: "bg-yellow-500/10 border-yellow-500",
    textClass: "text-yellow-600 dark:text-yellow-400",
    badgeClass: "bg-yellow-500 text-white",
    scoreBg: "bg-yellow-500",
  },
  "Legitimate": {
    icon: ShieldCheck,
    label: "Legitimate Message",
    bgClass: "bg-accent/10 border-accent",
    textClass: "text-accent",
    badgeClass: "bg-accent text-accent-foreground",
    scoreBg: "bg-accent",
  },
}

function detectLanguageLabel(text: string) {
  if (/[\u0900-\u097F]/.test(text)) return "Hindi detected"
  if (/[\u0B80-\u0BFF]/.test(text)) return "Tamil detected"
  if (/[\u0C00-\u0C7F]/.test(text)) return "Telugu detected"
  return "English detected"
}

const sampleMessages = [
  "आपका बैंक अकाउंट ब्लॉक हो गया है! तुरंत KYC अपडेट करें: bit.ly/verify-sbi",
  "உங்கள் OTP 834521. இதை யாரிடமும் பகிர வேண்டாம்.",
  "Hey, let's catch up for coffee tomorrow at 5pm?",
]

const placeholders = [
  { text: "Paste a suspicious message here...", lang: "English" },
  { text: "संदिग्ध संदेश यहाँ पेस्ट करें...", lang: "Hindi" },
  { text: "சந்தேகமான செய்தியை இங்கே ஒட்டவும்...", lang: "Tamil" },
  { text: "అనుమానాస్పద సందేశాన్ని ఇక్కడ పేస్ట్ చేయండి...", lang: "Telugu" },
  { text: "ಅನುಮಾನಾಸ್ಪದ ಸಂದೇಶವನ್ನು ಇಲ್ಲಿ ಅಂಟಿಸಿ...", lang: "Kannada" },
  { text: "സംശയാസ്പദമായ സന്ദേശം ഇവിടെ ഒട്ടിക്കുക...", lang: "Malayalam" },
  { text: "সন্দেহজনক বার্তা এখানে পেস্ট করুন...", lang: "Bengali" },
  { text: "શંકાસ્પદ સંદેશ અહીં પેસ્ટ કરો...", lang: "Gujarati" },
  { text: "ਸ਼ੱਕੀ ਸੁਨੇਹਾ ਇੱਥੇ ਪੇਸਟ ਕਰੋ...", lang: "Punjabi" },
  { text: "संशयास्पद संदेश इथे पेस्ट करा...", lang: "Marathi" },
]

export function HeroDemoInput() {
  const [message, setMessage] = useState("")
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [placeholderIndex, setPlaceholderIndex] = useState(0)
  const [displayText, setDisplayText] = useState("")
  const [isTyping, setIsTyping] = useState(true)

  const currentPlaceholder = placeholders[placeholderIndex]

  // Typing / erasing animation
  useEffect(() => {
    // Don't animate if user is typing
    if (message) return

    const target = currentPlaceholder.text
    let timeout: NodeJS.Timeout

    if (isTyping) {
      if (displayText.length < target.length) {
        timeout = setTimeout(() => {
          setDisplayText(target.slice(0, displayText.length + 1))
        }, 40)
      } else {
        // Pause at full text, then start erasing
        timeout = setTimeout(() => {
          setIsTyping(false)
        }, 1500)
      }
    } else {
      if (displayText.length > 0) {
        timeout = setTimeout(() => {
          setDisplayText(displayText.slice(0, -1))
        }, 25)
      } else {
        // Move to next language
        setPlaceholderIndex((prev) => (prev + 1) % placeholders.length)
        setIsTyping(true)
      }
    }

    return () => clearTimeout(timeout)
  }, [displayText, isTyping, message, currentPlaceholder.text])

  // Reset animation when switching placeholders
  const handleFocus = useCallback(() => {
    if (!message) {
      setDisplayText("")
      setIsTyping(true)
    }
  }, [message])

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
    <div id="demo" className="w-full">
      {/* Input card - styled like reference */}
      <div
        className="relative cursor-text overflow-hidden rounded-2xl border border-border bg-muted/40 backdrop-blur-sm transition-all focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/20"
        onClick={() => {
          const textarea = document.getElementById("hero-textarea")
          if (textarea) textarea.focus()
        }}
      >
        {/* Animated placeholder overlay (visible only when textarea is empty) */}
        {!message && (
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 px-6">
            <p className="text-balance text-center text-xl font-bold leading-snug text-foreground/70 md:text-2xl">
              {'"'}{displayText || currentPlaceholder.text}{'"'}
            </p>
            <span className="rounded-full bg-border px-3 py-1 text-xs font-medium text-muted-foreground">
              {currentPlaceholder.lang}
            </span>
          </div>
        )}

        <textarea
          id="hero-textarea"
          value={message}
          onChange={(e) => {
            setMessage(e.target.value)
            setResult(null)
          }}
          onFocus={handleFocus}
          rows={5}
          className="relative z-10 w-full resize-none bg-transparent px-6 pb-4 pt-8 text-center text-lg font-semibold text-foreground caret-primary placeholder:text-transparent focus:outline-none"
          placeholder=" "
        />

        {message.trim() && (
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            <Sparkles className="mr-1 inline h-3 w-3" />
            {detectLanguageLabel(message)}
          </div>
        )}
      </div>

      {/* Analyze button */}
      <Button
        onClick={handleAnalyze}
        disabled={!message.trim() || loading}
        className="mt-3 w-full bg-primary text-primary-foreground hover:bg-primary/85"
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
        <div className="mt-4 rounded-xl border-2 border-destructive/40 bg-destructive/5 p-4">
          <div className="flex items-center gap-2 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        </div>
      )}

      {/* Result */}
      {result && config && StatusIcon && (
        <div className={`mt-4 rounded-xl border-2 p-4 ${config.bgClass}`}>
          {/* Header: Status + Threat Score */}
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${config.badgeClass}`}>
                <StatusIcon className="h-4 w-4" />
              </div>
              <div>
                <p className={`text-sm font-bold ${config.textClass}`}>{config.label}</p>
                <p className="text-xs text-muted-foreground">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </p>
              </div>
            </div>
            {/* Threat Score */}
            <div className="flex flex-col items-center">
              <div className={`flex h-11 w-11 items-center justify-center rounded-full ${config.badgeClass}`}>
                <span className="text-base font-extrabold">{result.threat_score}</span>
              </div>
              <span className="mt-0.5 text-[10px] text-muted-foreground">Threat</span>
            </div>
          </div>

          {/* Summary */}
          {result.summary && (
            <p className="mb-3 text-xs font-medium text-foreground/80">{result.summary}</p>
          )}

          {/* Reasoning */}
          {result.reasoning && (
            <p className="mb-3 text-xs italic text-muted-foreground leading-relaxed">{result.reasoning}</p>
          )}

          {/* Detected signals */}
          {result.detected_signals && result.detected_signals.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1">
              {result.detected_signals.map((signal, i) => (
                <span key={i} className="rounded-full border border-border bg-muted/60 px-2 py-0.5 text-[10px] text-muted-foreground">
                  {signal.replace(/_/g, " ")}
                </span>
              ))}
            </div>
          )}

          {/* Model accuracy badge */}
          {result.model_accuracy != null && (
            <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/60 px-2.5 py-0.5 text-[11px] text-muted-foreground">
              <BrainCircuit className="h-3 w-3" />
              Model Accuracy: {(result.model_accuracy * 100).toFixed(1)}%
            </div>
          )}

          {/* Reasons */}
          <ul className="flex flex-col gap-1.5">
            {result.reasons.map((reason, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-foreground">
                {result.classification === "Highly Likely Phishing" ? (
                  <AlertTriangle className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${config.textClass}`} />
                ) : result.classification === "Suspicious" ? (
                  <AlertTriangle className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${config.textClass}`} />
                ) : (
                  <CircleCheck className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${config.textClass}`} />
                )}
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Sample messages */}
      <div className="mt-4">
        <p className="mb-2 text-[11px] font-medium text-muted-foreground">Try a sample:</p>
        <div className="flex flex-wrap gap-1.5">
          {sampleMessages.map((sample) => (
            <button
              key={sample}
              onClick={() => {
                setMessage(sample)
                setResult(null)
              }}
              className="rounded-lg border border-border bg-muted/50 px-2.5 py-1 text-left text-[11px] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {sample.length > 40 ? sample.slice(0, 40) + "..." : sample}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
