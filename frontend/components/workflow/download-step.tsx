"use client"

import { Button } from "@/components/ui/button"
import { Download, RotateCcw, Check, Box, FileCode, Layers } from "lucide-react"
import { Model3DViewer } from "@/components/model-3d-viewer"

interface DownloadStepProps {
  objFilename: string | null
  onDownload: () => void
  onReset: () => void
  apiBaseUrl: string
}

export function DownloadStep({ objFilename, onDownload, onReset, apiBaseUrl }: DownloadStepProps) {
  return (
    <div className="rounded-2xl border border-border bg-card p-8">
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Step 4: Download Your 3D Model</h3>
        <p className="text-muted-foreground">Your 3D model is ready! Preview it below or download the OBJ file.</p>
      </div>

      <div className="mb-6 p-4 rounded-xl bg-success/10 border border-success/30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-success/20 flex items-center justify-center">
            <Check className="w-4 h-4 text-success" />
          </div>
          <div>
            <p className="font-medium text-success">3D Model Generated Successfully!</p>
            <p className="text-xs text-muted-foreground">Your floor plan has been converted to a 3D OBJ file</p>
          </div>
        </div>
      </div>

      {/* 3D Viewer */}
      <div className="mb-6 rounded-xl overflow-hidden border border-border bg-secondary/30 h-[400px]">
        <Model3DViewer objUrl={objFilename ? `${apiBaseUrl}/download/${objFilename}` : null} />
      </div>

      {/* File Info */}
      <div className="grid sm:grid-cols-3 gap-4 mb-6">
        <div className="p-4 rounded-xl bg-secondary/50 border border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <FileCode className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">File Name</p>
              <p className="font-medium text-sm truncate">{objFilename}</p>
            </div>
          </div>
        </div>
        <div className="p-4 rounded-xl bg-secondary/50 border border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Box className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Format</p>
              <p className="font-medium text-sm">Wavefront OBJ</p>
            </div>
          </div>
        </div>
        <div className="p-4 rounded-xl bg-secondary/50 border border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Layers className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Contains</p>
              <p className="font-medium text-sm">Geometry + Colors</p>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row items-center gap-4">
        <Button onClick={onDownload} className="w-full sm:w-auto flex-1" size="lg">
          <Download className="w-4 h-4 mr-2" />
          Download 3D Model (.obj)
        </Button>
        <Button variant="outline" onClick={onReset} size="lg" className="w-full sm:w-auto bg-transparent">
          <RotateCcw className="w-4 h-4 mr-2" />
          Process Another Floor Plan
        </Button>
      </div>

      {/* Usage Tips */}
      <div className="mt-6 p-4 rounded-xl bg-primary/5 border border-primary/20">
        <h4 className="font-medium text-sm text-primary mb-2">Recommended 3D Software</h4>
        <p className="text-xs text-muted-foreground">
          Open your OBJ file in Blender, SketchUp, AutoCAD, 3ds Max, or any OBJ-compatible software. The model includes
          vertex colors for walls (gray), doors (brown), windows (blue), and floors (light gray).
        </p>
      </div>
    </div>
  )
}
