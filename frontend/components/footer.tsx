import { Shield, Github, Linkedin } from "lucide-react"

const quickLinks = [
  { label: "How It Works", href: "#how-it-works" },
  { label: "Languages", href: "#languages" },
  { label: "Why SurakshaAI", href: "#why" },
]

export function Footer() {
  return (
    <footer className="border-t border-border bg-card">
      <div className="mx-auto max-w-7xl px-4 py-12 lg:px-8">
        <div className="grid gap-10 md:grid-cols-2">
          {/* Brand */}
          <div>
            <a href="#" className="mb-4 flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
                <Shield className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">
                Suraksha<span className="text-primary">AI</span>
              </span>
            </a>
            <p className="mb-4 max-w-sm text-sm leading-relaxed text-muted-foreground">
              AI-powered multilingual phishing detection built in India, for India. Protecting families from digital
              scams across every language.
            </p>
            {/* Indian flag accent */}
            <div className="flex items-center gap-1.5">
              <div className="h-2 w-6 rounded-full bg-orange-400" />
              <div className="h-2 w-6 rounded-full border border-border bg-white" />
              <div className="h-2 w-6 rounded-full bg-green-500" />
              <span className="ml-2 text-xs text-muted-foreground">Made in India</span>
            </div>
          </div>

          {/* Better right panel */}
          <div className="rounded-2xl border border-border bg-background/40 p-6">
            <h4 className="mb-3 text-sm font-semibold text-foreground">Quick Links</h4>
            <ul className="mb-6 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {quickLinks.map((link) => (
                <li key={link.label}>
                  <a
                    href={link.href}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>

            <h4 className="mb-3 text-sm font-semibold text-foreground">Connect</h4>
            <div className="flex flex-wrap gap-3">
              <a
                href="https://www.linkedin.com/in/anuroop-phukan/"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                <Linkedin className="h-4 w-4" /> LinkedIn
              </a>
              <a
                href="https://github.com/Anuroop-cloud/phishbait"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                <Github className="h-4 w-4" /> GitHub
              </a>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-10 border-t border-border pt-6">
          <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
            <p className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} SurakshaAI. All rights reserved.
            </p>
            <p className="text-xs text-muted-foreground">Built with care during a 36-hour hackathon.</p>
          </div>
        </div>
      </div>
    </footer>
  )
}
