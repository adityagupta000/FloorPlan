"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, ImageIcon, X, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface UploadStepProps {
  onUploadComplete: (filename: string, imageUrl: string) => void
  isProcessing: boolean
  setProcessing: (processing: boolean) => void
  setError: (error: string | null) => void
  apiBaseUrl: string
}

export function UploadStep({ onUploadComplete, isProcessing, setProcessing, setError, apiBaseUrl }: UploadStepProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (file) {
        // Validate file type
        const validTypes = ["image/png", "image/jpeg", "image/jpg"]
        if (!validTypes.includes(file.type)) {
          setError("Invalid file type. Please upload PNG, JPG, or JPEG.")
          return
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
          setError("File too large. Maximum size is 16MB.")
          return
        }

        setSelectedFile(file)
        setPreview(URL.createObjectURL(file))
        setError(null)
      }
    },
    [setError],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
    },
    maxFiles: 1,
    disabled: isProcessing,
  })

  const clearSelection = () => {
    setSelectedFile(null)
    setPreview(null)
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setProcessing(true)
    setError(null)

    const formData = new FormData()
    formData.append("image", selectedFile)

    try {
      const response = await fetch(`${apiBaseUrl}/upload`, {
        method: "POST",
        body: formData,
      })

      const data = await response.json()

      if (response.ok) {
        onUploadComplete(data.filename, preview!)
      } else {
        throw new Error(data.error || "Upload failed")
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Upload failed")
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="rounded-2xl border border-border bg-card p-8">
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Step 1: Upload Floor Plan</h3>
        <p className="text-muted-foreground">
          Upload your 2D floor plan image. Supported formats: PNG, JPG, JPEG (max 16MB)
        </p>
      </div>

      {!preview ? (
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200",
            isDragActive
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-secondary/50",
            isProcessing && "opacity-50 cursor-not-allowed",
          )}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
              <Upload className="w-8 h-8 text-primary" />
            </div>
            <div>
              <p className="text-lg font-medium">
                {isDragActive ? "Drop your floor plan here" : "Drag & drop your floor plan"}
              </p>
              <p className="text-sm text-muted-foreground mt-1">or click to browse files</p>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <ImageIcon className="w-4 h-4" />
              <span>PNG, JPG, JPEG up to 16MB</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="relative rounded-xl overflow-hidden border border-border bg-secondary/30">
            <img
              src={preview || "/placeholder.svg"}
              alt="Floor plan preview"
              className="w-full h-auto max-h-[400px] object-contain"
            />
            <button
              onClick={clearSelection}
              className="absolute top-4 right-4 p-2 rounded-full bg-background/80 backdrop-blur-sm hover:bg-background transition-colors"
              disabled={isProcessing}
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <ImageIcon className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="font-medium text-sm">{selectedFile?.name}</p>
                <p className="text-xs text-muted-foreground">
                  {selectedFile && (selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Button variant="outline" onClick={clearSelection} disabled={isProcessing}>
                Change File
              </Button>
              <Button onClick={handleUpload} disabled={isProcessing} className="bg-primary hover:bg-primary/90">
                {isProcessing ? (
                  <>
                    <span className="animate-spin mr-2"></span>
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload & Continue
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Info Box */}
      <div className="mt-6 p-4 rounded-xl bg-primary/5 border border-primary/20">
        <div className="flex gap-3">
          <AlertCircle className="w-5 h-5 text-primary shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-primary">Tips for best results</p>
            <ul className="mt-2 text-xs text-muted-foreground space-y-1">
              <li>• Use high-resolution images for better segmentation</li>
              <li>• Ensure walls, doors, and windows are clearly visible</li>
              <li>• Black and white or colored floor plans both work</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
