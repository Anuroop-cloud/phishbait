import { Navbar } from "@/components/navbar"
import { HeroSection } from "@/components/hero-section"
import { LiveDemo } from "@/components/live-demo"
import { HowItWorks } from "@/components/how-it-works"
import { MultilingualSection } from "@/components/multilingual-section"
import { WhySuraksha } from "@/components/why-suraksha"
import { CommunitySection } from "@/components/community-section"
import { Footer } from "@/components/footer"

export default function Page() {
  return (
    <main>
      <Navbar />
      <HeroSection />
      <LiveDemo />
      <HowItWorks />
      <MultilingualSection />
      <WhySuraksha />
      <CommunitySection />
      <Footer />
    </main>
  )
}
