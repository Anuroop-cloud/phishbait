import { BrainCircuit, Layers, ScanSearch } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const steps = [
  {
    icon: BrainCircuit,
    title: "Intelligent Text Processing",
    description:
      "Our AI model understands Indian languages contextually using IndicBERT, a transformer-based model trained on 12+ Indian languages.",
    highlights: ["Contextual understanding", "Transformer-based (IndicBERT)", "12+ Indian languages"],
    color: "bg-primary/20 text-primary",
  },
  {
    icon: ScanSearch,
    title: "Multi-Layer Feature Analysis",
    description:
      'Detects suspicious URLs, urgency words like "तुरंत" and "உடனே", fake authority impersonation, and structural anomalies in messages.',
    highlights: ["Suspicious URL detection", "Urgency language patterns", "Authority impersonation"],
    color: "bg-secondary/20 text-secondary",
  },
  {
    icon: Layers,
    title: "Hybrid AI Classification",
    description:
      "Combines semantic embeddings with handcrafted phishing features using an ensemble classifier for explainable, accurate results.",
    highlights: ["Semantic embeddings", "Ensemble classifier", "80-90+% accuracy"],
    color: "bg-accent/20 text-accent",
  },
]

export function HowItWorks() {
  return (
    <section id="how-it-works" className="bg-muted py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 lg:px-8">
        <div className="mx-auto mb-14 max-w-2xl text-center">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wider text-primary">
            How It Works
          </p>
          <h2 className="mb-4 text-balance text-3xl font-bold text-foreground md:text-4xl">
            Three Layers of AI Protection
          </h2>
          <p className="text-pretty text-lg text-muted-foreground">
            Our hybrid approach combines the power of deep learning with traditional feature engineering for maximum
            accuracy.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {steps.map((step, i) => (
            <Card
              key={step.title}
              className="group relative overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-shadow duration-300 hover:shadow-lg"
            >
              {/* Step number */}
              <div className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-full bg-border text-sm font-bold text-muted-foreground">
                {i + 1}
              </div>

              <CardHeader className="pb-2">
                <div className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl ${step.color}`}>
                  <step.icon className="h-6 w-6" />
                </div>
                <CardTitle className="text-xl text-card-foreground">{step.title}</CardTitle>
              </CardHeader>

              <CardContent>
                <p className="mb-4 leading-relaxed text-muted-foreground">{step.description}</p>
                <ul className="flex flex-col gap-2">
                  {step.highlights.map((h) => (
                    <li key={h} className="flex items-center gap-2 text-sm text-foreground">
                      <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                      {h}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
