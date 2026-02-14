"use client"

import { useState } from "react"
import { Shield, Menu, X } from "lucide-react"

const navLinks = [
  { label: "Home", href: "/" },
  { label: "How It Works", href: "/how-it-works#how-it-works" },
  { label: "Languages", href: "/languages#languages" },
  { label: "Community", href: "/#community" },
]

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/90 backdrop-blur-md supports-[backdrop-filter]:bg-background/80">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 lg:px-8">
        {/* Logo */}
        <a href="/" className="group flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary transition-all duration-200 group-hover:animate-pop-dramatic group-hover:shadow-[0_0_24px_rgba(16,185,129,0.45)]">
            <Shield className="h-5 w-5 text-primary-foreground" />
          </div>
          <span className="text-xl font-bold text-foreground transition-all duration-200 group-hover:animate-pop-dramatic">
            Suraksha<span className="text-primary">AI</span>
          </span>
        </a>

        {/* Desktop links */}
        <ul className="hidden items-center gap-6 lg:flex">
          {navLinks.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="inline-flex items-center rounded-lg px-2 py-1 text-sm font-medium text-muted-foreground transition-all duration-200 hover:animate-pop-dramatic hover:text-foreground hover:shadow-[0_0_24px_rgba(16,185,129,0.35)] active:scale-95"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        {/* Mobile toggle */}
        <button
          className="lg:hidden"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label={mobileOpen ? "Close menu" : "Open menu"}
        >
          {mobileOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </nav>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="border-t border-border bg-background px-4 pb-4 lg:hidden">
          <ul className="flex flex-col gap-3 py-3">
            {navLinks.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  className="block rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-all duration-200 hover:animate-pop-dramatic hover:bg-muted hover:text-foreground hover:shadow-[0_0_24px_rgba(16,185,129,0.28)] active:scale-95"
                  onClick={() => setMobileOpen(false)}
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </header>
  )
}
