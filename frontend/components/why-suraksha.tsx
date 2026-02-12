import { Target, BrainCircuit, Fingerprint, ShieldAlert, Landmark, Phone, CreditCard, KeyRound } from "lucide-react"

const stats = [
  { icon: Target, label: "Detection Accuracy", value: "90%" },
  { icon: BrainCircuit, label: "Hybrid AI Architecture", value: "IndicBERT+" },
  { icon: Fingerprint, label: "Built for India", value: "12+ Languages" },
]

const scamTypes = [
  { icon: Landmark, label: "Bank Scams", description: "Fake alerts from SBI, HDFC, and other banks" },
  { icon: ShieldAlert, label: "Digital Arrest Scams", description: "Impersonation of police or CBI officials" },
  { icon: CreditCard, label: "KYC Fraud", description: "Fraudulent Aadhaar and PAN verification requests" },
  { icon: KeyRound, label: "OTP Scams", description: "Tricks to steal one-time passwords" },
  { icon: Phone, label: "UPI Fraud", description: "Fake UPI payment requests and links" },
]

export function WhySuraksha() {
  return (
    <section id="why" className="mandala-bg-subtle py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 lg:px-8">
        <div className="mx-auto mb-14 max-w-2xl text-center">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wider text-primary">
            Why SurakshaAI
          </p>
          <h2 className="mb-4 text-balance text-3xl font-bold text-foreground md:text-4xl">
            India&apos;s Shield Against Cyber Scams
          </h2>
          <p className="text-pretty text-lg text-muted-foreground">
            Purpose-built for the Indian digital ecosystem with world-class AI accuracy.
          </p>
        </div>

        {/* Stats */}
        <div className="mx-auto mb-16 grid max-w-3xl grid-cols-1 gap-4 sm:grid-cols-3">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="flex flex-col items-center rounded-2xl border border-border bg-card p-6 text-center shadow-sm"
            >
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/20 text-primary">
                <stat.icon className="h-6 w-6" />
              </div>
              <p className="text-3xl font-extrabold text-foreground">{stat.value}</p>
              <p className="text-sm text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Scam types */}
        <div className="mx-auto max-w-4xl">
          <h3 className="mb-6 text-center text-xl font-bold text-foreground">Scam Types We Detect</h3>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {scamTypes.map((scam) => (
              <div
                key={scam.label}
                className="flex items-start gap-4 rounded-xl border border-border bg-card p-4 shadow-sm transition-shadow hover:shadow-md"
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-destructive/15 text-destructive">
                  <scam.icon className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-semibold text-foreground">{scam.label}</p>
                  <p className="text-sm leading-relaxed text-muted-foreground">{scam.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
