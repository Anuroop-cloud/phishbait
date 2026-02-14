import { Navbar } from "@/components/navbar"
import { HeroSection } from "@/components/hero-section"
import { WhySuraksha } from "@/components/why-suraksha"
import { CommunitySection } from "@/components/community-section"
import { Footer } from "@/components/footer"

export default function Page() {
  return (
    <main>
      <Navbar />
      <HeroSection />
      <WhySuraksha />
      <CommunitySection />
      <Footer />
    </main>
  )
}
