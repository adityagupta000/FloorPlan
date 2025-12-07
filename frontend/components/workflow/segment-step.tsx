"use client";

import { Button } from "@/components/ui/button";
import { Cpu, ArrowLeft, ImageIcon } from "lucide-react";

interface SegmentStepProps {
  imageUrl: string | null;
  filename: string | null;
  onSegment: () => void;
  onBack: () => void;
  isProcessing: boolean;
}

export function SegmentStep({
  imageUrl,
  filename,
  onSegment,
  onBack,
  isProcessing,
}: SegmentStepProps) {
  return (
    <div className="rounded-2xl border border-border bg-card p-8">
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Step 2: AI Segmentation</h3>
        <p className="text-muted-foreground">
          Our AI model will analyze your floor plan and identify walls, doors,
          windows, and floor areas.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Preview */}
        <div className="rounded-xl overflow-hidden border border-border bg-secondary/30">
          {imageUrl && (
            <img
              src={imageUrl || "/placeholder.svg"}
              alt="Uploaded floor plan"
              className="w-full h-auto max-h-[300px] object-contain"
            />
          )}
        </div>

        {/* Info Panel */}
        <div className="flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <ImageIcon className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="font-medium text-sm">{filename}</p>
                <p className="text-xs text-muted-foreground">
                  Ready for segmentation
                </p>
              </div>
            </div>

            <div className="space-y-3 p-4 rounded-xl bg-secondary/50 border border-border">
              <h4 className="font-medium text-sm">Segmentation Classes:</h4>
              <div className="grid grid-cols-2 gap-2">
                {/* Walls */}
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: "#B4B4B4" }}
                  />
                  <span className="text-xs">Walls</span>
                </div>

                {/* Doors */}
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: "#A5372D" }}
                  />
                  <span className="text-xs">Doors</span>
                </div>

                {/* Windows (glass color used here with opacity) */}
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    // #32AAFF -> rgba(50,170,255,0.6)
                    style={{ backgroundColor: "rgba(50,170,255,0.6)" }}
                  />
                  <span className="text-xs">Windows (Glass)</span>
                </div>

                {/* Window Frames (use the frame color) */}
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: "#FFB41E" }}
                  />
                  <span className="text-xs">Window Frames</span>
                </div>

                {/* Floors */}
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: "#D2BE96" }}
                  />
                  <span className="text-xs">Floors</span>
                </div>
              </div>
            </div>

            <div className="mt-4 p-4 rounded-xl bg-primary/5 border border-primary/20">
              <p className="text-xs text-muted-foreground">
                <strong className="text-primary">Model:</strong> ResNet50-UNet
                with Attention Gates
                <br />
                <strong className="text-primary">Output:</strong> 512×512
                segmentation mask
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 mt-6">
            <Button variant="outline" onClick={onBack} disabled={isProcessing}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <Button
              onClick={onSegment}
              disabled={isProcessing}
              className="flex-1 bg-primary hover:bg-primary/90"
            >
              {isProcessing ? (
                <>
                  <span className="animate-spin mr-2"></span>
                  Running Segmentation...
                </>
              ) : (
                <>
                  <Cpu className="w-4 h-4 mr-2" />
                  Run AI Segmentation
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
              <p className="font-medium">Processing your floor plan...</p>
              <p className="text-sm text-muted-foreground">
                This may take a few seconds
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
