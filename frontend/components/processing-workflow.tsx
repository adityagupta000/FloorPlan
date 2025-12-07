"use client"

import { useState, useCallback } from "react"
import { UploadStep } from "@/components/workflow/upload-step"
import { SegmentStep } from "@/components/workflow/segment-step"
import { GenerateStep } from "@/components/workflow/generate-step"
import { DownloadStep } from "@/components/workflow/download-step"
import { StepIndicator } from "@/components/workflow/step-indicator"

export type WorkflowStep = 1 | 2 | 3 | 4

export interface WorkflowState {
  currentStep: WorkflowStep
  uploadedFilename: string | null
  uploadedImageUrl: string | null
  maskFilename: string | null
  objFilename: string | null
  isProcessing: boolean
  error: string | null
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api"

export function ProcessingWorkflow() {
  const [state, setState] = useState<WorkflowState>({
    currentStep: 1,
    uploadedFilename: null,
    uploadedImageUrl: null,
    maskFilename: null,
    objFilename: null,
    isProcessing: false,
    error: null,
  })

  const setError = useCallback((error: string | null) => {
    setState((prev) => ({ ...prev, error }))
  }, [])

  const setProcessing = useCallback((isProcessing: boolean) => {
    setState((prev) => ({ ...prev, isProcessing }))
  }, [])

  const handleUploadComplete = useCallback((filename: string, imageUrl: string) => {
    setState((prev) => ({
      ...prev,
      uploadedFilename: filename,
      uploadedImageUrl: imageUrl,
      currentStep: 2,
      error: null,
    }))
  }, [])

  const handleSegmentComplete = useCallback((maskFilename: string) => {
    setState((prev) => ({
      ...prev,
      maskFilename,
      currentStep: 3,
      error: null,
    }))
  }, [])

  const handleGenerateComplete = useCallback((objFilename: string) => {
    setState((prev) => ({
      ...prev,
      objFilename,
      currentStep: 4,
      error: null,
    }))
  }, [])

  const handleReset = useCallback(() => {
    setState({
      currentStep: 1,
      uploadedFilename: null,
      uploadedImageUrl: null,
      maskFilename: null,
      objFilename: null,
      isProcessing: false,
      error: null,
    })
  }, [])

  const runSegmentation = async () => {
    if (!state.uploadedFilename) return

    setProcessing(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/segment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: state.uploadedFilename }),
      })

      const data = await response.json()

      if (response.ok) {
        handleSegmentComplete(data.mask_filename)
      } else {
        throw new Error(data.error || "Segmentation failed")
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Segmentation failed")
    } finally {
      setProcessing(false)
    }
  }

  const generate3DModel = async () => {
    if (!state.maskFilename) return

    setProcessing(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/generate3d`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask_filename: state.maskFilename }),
      })

      const data = await response.json()

      if (response.ok) {
        handleGenerateComplete(data.obj_filename)
      } else {
        throw new Error(data.error || "3D generation failed")
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "3D generation failed")
    } finally {
      setProcessing(false)
    }
  }

  const downloadFile = async () => {
    if (!state.objFilename) return

    try {
      const response = await fetch(`${API_BASE_URL}/download/${state.objFilename}`)

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = state.objFilename
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      } else {
        throw new Error("Download failed")
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Download failed")
    }
  }

  return (
    <section id="workflow" className="py-20 bg-secondary/30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-12">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Convert Your Floor Plan</h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Follow these simple steps to transform your 2D floor plan into a detailed 3D model
          </p>
        </div>

        {/* Step Indicator */}
        <StepIndicator currentStep={state.currentStep} />

        {/* Error Display */}
        {state.error && (
          <div className="mb-8 p-4 rounded-xl border border-destructive/50 bg-destructive/10 text-destructive max-w-4xl mx-auto">
            <p className="text-sm font-medium">{state.error}</p>
            <button onClick={() => setError(null)} className="mt-2 text-xs underline hover:no-underline">
              Dismiss
            </button>
          </div>
        )}

        {/* Workflow Steps */}
        <div className="max-w-4xl mx-auto">
          {state.currentStep === 1 && (
            <UploadStep
              onUploadComplete={handleUploadComplete}
              isProcessing={state.isProcessing}
              setProcessing={setProcessing}
              setError={setError}
              apiBaseUrl={API_BASE_URL}
            />
          )}

          {state.currentStep === 2 && (
            <SegmentStep
              imageUrl={state.uploadedImageUrl}
              filename={state.uploadedFilename}
              onSegment={runSegmentation}
              onBack={handleReset}
              isProcessing={state.isProcessing}
            />
          )}

          {state.currentStep === 3 && (
            <GenerateStep
              imageUrl={state.uploadedImageUrl}
              onGenerate={generate3DModel}
              onBack={handleReset}
              isProcessing={state.isProcessing}
            />
          )}

          {state.currentStep === 4 && (
            <DownloadStep
              objFilename={state.objFilename}
              onDownload={downloadFile}
              onReset={handleReset}
              apiBaseUrl={API_BASE_URL}
            />
          )}
        </div>
      </div>
    </section>
  )
}
