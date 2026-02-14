import Image from "next/image"
import { Button } from "@/components/ui/button"
import { ShieldCheck, ArrowRight } from "lucide-react"
import { HeroDemoInput } from "@/components/hero-demo-input"

export function HeroSection() {
  return (
    <section className="mandala-bg relative overflow-hidden">
      <div className="mx-auto flex max-w-7xl flex-col-reverse items-center gap-10 px-4 py-16 lg:flex-row lg:items-start lg:gap-12 lg:px-8 lg:py-24">
        {/* Text content */}
        <div className="flex-1 text-center lg:text-left">
          <div className="animate-fade-in-up mb-4 inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary">
            <ShieldCheck className="h-4 w-4" />
            AI-Powered Cyber Protection
          </div>

          <h1 className="animate-fade-in-up-delay-1 mb-6 text-balance text-4xl font-extrabold leading-tight tracking-tight text-foreground md:text-5xl lg:text-6xl">
            Protecting India From Digital Scams{" "}
            <span className="text-primary">{'— In Every Language.'}</span>
          </h1>

          <p className="animate-fade-in-up-delay-2 mx-auto mb-8 max-w-xl text-pretty text-lg leading-relaxed text-muted-foreground lg:mx-0">
            SurakshaAI uses advanced transformer-based AI (IndicBERT) and hybrid feature analysis to detect phishing
            messages across Hindi, Tamil, Telugu, and more Indian languages.
          </p>

          <div className="animate-fade-in-up-delay-3 flex flex-col items-center gap-3 sm:flex-row lg:justify-start">
            <Button
              asChild
              size="lg"
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              <a href="#demo" className="gap-2">
                Check a Message
                <ArrowRight className="h-4 w-4" />
              </a>
            </Button>
            <Button asChild size="lg" variant="outline" className="border-border text-foreground hover:bg-muted">
              <a href="/languages#how-it-works">How It Works</a>
            </Button>
          </div>
        </div>

        {/* Illustration */}
        <div className="animate-fade-in-up flex-1">
          <div className="mx-auto w-full max-w-md lg:max-w-lg">
            <div className="relative mb-6">
              <div className="absolute -inset-4 rounded-3xl bg-primary/15 blur-2xl" />
              <Image
                src="/hero-illustration.jpg"
                alt="Illustration of an Indian woman checking her phone for suspicious messages"
                width={520}
                height={520}
                className="relative z-10 rounded-2xl"
                priority
              />
            </div>
          </div>
        </div>
      </div>

      {/* Full-width Demo Input */}
      <div className="relative z-10 mx-auto w-full max-w-7xl px-4 pb-16 lg:px-8">
        <HeroDemoInput />
      </div>

      {/* Pattern divider */}
      <div className="pattern-divider" />
    </section>
  )
}
