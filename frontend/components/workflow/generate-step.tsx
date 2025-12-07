"use client"

import { Button } from "@/components/ui/button"
import { Box, ArrowLeft, Check } from "lucide-react"

interface GenerateStepProps {
  imageUrl: string | null
  onGenerate: () => void
  onBack: () => void
  isProcessing: boolean
}

export function GenerateStep({ imageUrl, onGenerate, onBack, isProcessing }: GenerateStepProps) {
  return (
    <div className="rounded-2xl border border-border bg-card p-8">
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Step 3: Generate 3D Model</h3>
        <p className="text-muted-foreground">
          Segmentation complete! Now let&apos;s convert the mask into a detailed 3D OBJ model.
        </p>
      </div>

      {/* Success Banner */}
      <div className="mb-6 p-4 rounded-xl bg-green-500/10 border border-green-500/30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
            <Check className="w-4 h-4 text-green-500" />
          </div>
          <div>
            <p className="font-medium text-green-500">Segmentation Complete</p>
            <p className="text-xs text-muted-foreground">Detected walls, doors, windows, and floor regions</p>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Preview */}
        <div className="rounded-xl overflow-hidden border border-border bg-secondary/30">
          {imageUrl && (
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="Floor plan"
              className="w-full h-auto max-h-[300px] object-contain"
            />
          )}
        </div>

        {/* Info Panel */}
        <div className="flex flex-col justify-between">
          <div className="space-y-4 p-4 rounded-xl bg-secondary/50 border border-border">
            <h4 className="font-medium">3D Model Details:</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-gray-400" />
                Walls with proper openings (0.8m height)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-amber-600" />
                Door frames at floor level
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                Window frames at realistic height
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-yellow-400" />
                Floor slab (2cm thick)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary" />
                Scale: 1px = 1cm
              </li>
            </ul>
          </div>

          <div className="flex items-center gap-3 mt-6">
            <Button variant="outline" onClick={onBack} disabled={isProcessing}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Start Over
            </Button>
            <Button onClick={onGenerate} disabled={isProcessing} className="flex-1 bg-primary hover:bg-primary/90">
              {isProcessing ? (
                <>
                  <span className="animate-spin mr-2"></span>
                  Generating 3D Model...
                </>
              ) : (
                <>
                  <Box className="w-4 h-4 mr-2" />
                  Generate 3D Model
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Processing Animation */}
      {isProcessing && (
        <div className="mt-6 p-6 rounded-xl bg-secondary/50 border border-border">
          <div className="flex items-center justify-center gap-4">
            <div className="relative w-12 h-12">
              <div className="absolute inset-0 rounded-full border-2 border-primary/30" />
              <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-primary animate-spin" />
            </div>
            <div>
              <p className="font-medium">Generating 3D geometry...</p>
              <p className="text-sm text-muted-foreground">Creating vertices, faces, and materials</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
