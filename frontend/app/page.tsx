import { Header } from "@/components/header"
import { Hero } from "@/components/hero"
import { ProcessingWorkflow } from "@/components/processing-workflow"
import { Features } from "@/components/features"
import { Footer } from "@/components/footer"

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <Hero />
        <ProcessingWorkflow />
        <Features />
      </main>
      <Footer />
    </div>
  )
}
