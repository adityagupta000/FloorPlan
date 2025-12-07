"use client"

import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles, Layers, Zap } from "lucide-react"

export function Hero() {
  const scrollToWorkflow = () => {
    document.getElementById("workflow")?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <section className="relative pt-32 pb-20 overflow-hidden">
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 grid-pattern opacity-30" />

      <div className="absolute top-20 left-1/4 w-96 h-96 bg-primary/15 rounded-full blur-3xl" />
      <div className="absolute bottom-20 right-1/4 w-80 h-80 bg-primary/10 rounded-full blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          {/* <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-secondary/50 mb-8">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">AI-Powered Precision</span>
          </div> */}

          {/* Main Heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tight text-balance max-w-4xl mx-auto">
            Transform Floor Plans
            <br />
            <span className="text-primary">Into 3D Models</span>
          </h1>

          {/* Subheading */}
          <p className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto text-pretty">
            Upload your 2D floor plan and watch as our AI segments walls, doors, windows, and floors to generate a
            detailed 3D OBJ model in seconds.
          </p>

          {/* CTAs */}
          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button size="lg" className="px-8 gap-2" onClick={scrollToWorkflow}>
              Start Converting
              <ArrowRight className="h-4 w-4" />
            </Button>
            {/* <Button size="lg" variant="outline" className="gap-2 bg-transparent">
              View Documentation
            </Button> */}
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-8 max-w-3xl mx-auto">
            <div className="flex flex-col items-center p-6 rounded-xl border border-border bg-card/50 hover:border-primary/50 transition-colors">
              <Layers className="h-8 w-8 text-primary mb-3" />
              <span className="text-2xl font-bold">5 Classes</span>
              <span className="text-sm text-muted-foreground">Semantic Segmentation</span>
            </div>
            <div className="flex flex-col items-center p-6 rounded-xl border border-border bg-card/50 hover:border-primary/50 transition-colors">
              <Zap className="h-8 w-8 text-primary mb-3" />
              <span className="text-2xl font-bold">&lt;30s</span>
              <span className="text-sm text-muted-foreground">Processing Time</span>
            </div>
            <div className="flex flex-col items-center p-6 rounded-xl border border-border bg-card/50 hover:border-primary/50 transition-colors">
              <Sparkles className="h-8 w-8 text-primary mb-3" />
              <span className="text-2xl font-bold">ResNet50</span>
              <span className="text-sm text-muted-foreground">UNet Architecture</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
