import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'

import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'SurakshaAI — Protecting India From Digital Scams',
  description:
    'AI-powered multilingual phishing detection platform that protects Indian users from cyber scams in Hindi, Tamil, Telugu, and more.',
  keywords: ['cybersecurity', 'phishing detection', 'India', 'AI', 'multilingual', 'scam protection'],
}

export const viewport: Viewport = {
  themeColor: '#000080',
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}
