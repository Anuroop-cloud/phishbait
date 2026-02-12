import { Heart, CheckCircle2 } from "lucide-react"

const tips = [
  "Never share OTP or PIN with anyone, even if they claim to be from your bank.",
  "Government agencies will never ask for money over phone or video call.",
  "Check URLs carefully \u2014 scammers use misspelled bank names.",
  "If a message creates panic, it\u2019s likely a scam. Take a breath and verify.",
]

export function CommunitySection() {
  return (
    <section id="community" className="bg-muted py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 lg:px-8">
        <div className="mx-auto max-w-4xl">
          <div className="grid items-center gap-10 lg:grid-cols-2">
            {/* Left: text + tips */}
            <div>
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary">
                <Heart className="h-4 w-4 text-destructive" />
                Community Awareness
              </div>
              <h2 className="mb-4 text-balance text-3xl font-bold text-foreground md:text-4xl">
                Teach Your Parents & Grandparents to Stay Safe
              </h2>
              <p className="mb-6 text-pretty text-lg leading-relaxed text-muted-foreground">
                Cyber scams target those who are less tech-savvy. Share these simple tips with your family to keep them
                protected.
              </p>

              <ul className="flex flex-col gap-3">
                {tips.map((tip) => (
                  <li key={tip} className="flex items-start gap-3">
                    <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-accent" />
                    <span className="text-sm leading-relaxed text-foreground">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Right: WhatsApp-style mockup */}
            <div className="flex justify-center">
              <div className="w-full max-w-sm rounded-2xl border border-border bg-card p-4 shadow-lg">
                {/* Chat header */}
                <div className="mb-4 flex items-center gap-3 border-b border-border pb-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-accent text-accent-foreground text-sm font-bold">
                    SA
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-foreground">SurakshaAI Bot</p>
                    <p className="text-xs text-muted-foreground">Online</p>
                  </div>
                </div>

                {/* Messages */}
                <div className="flex flex-col gap-3">
                  {/* Scam message */}
                  <div className="self-start rounded-2xl rounded-tl-md bg-muted px-4 py-2.5">
                    <p className="text-sm text-foreground">
                      {'आपका SBI अकाउंट ब्लॉक हो गया है! तुरंत KYC अपडेट करें '}
                      <span className="text-primary underline">bit.ly/sbi-kyc</span>
                    </p>
                    <p className="mt-1 text-[10px] text-muted-foreground">10:32 AM</p>
                  </div>

                  {/* Bot analysis */}
                  <div className="self-end rounded-2xl rounded-tr-md bg-primary px-4 py-2.5">
                    <p className="text-sm font-medium text-primary-foreground">
                      {'Highly Likely Phishing! This message is fake.'}
                    </p>
                    <p className="mt-1 text-xs text-primary-foreground/70">
                      {'Suspicious URL + Urgency language + Bank impersonation detected.'}
                    </p>
                    <p className="mt-1 text-[10px] text-primary-foreground/50">10:32 AM</p>
                  </div>

                  {/* User reply */}
                  <div className="self-start rounded-2xl rounded-tl-md bg-muted px-4 py-2.5">
                    <p className="text-sm text-foreground">{'Thank you! Almost clicked the link 😅'}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">10:33 AM</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
