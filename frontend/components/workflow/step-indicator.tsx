"use client"

import { Check, Upload, Cpu, Box, Download } from "lucide-react"
import { cn } from "@/lib/utils"
import type { WorkflowStep } from "@/components/processing-workflow"

interface StepIndicatorProps {
  currentStep: WorkflowStep
}

const steps = [
  { number: 1, label: "Upload", icon: Upload },
  { number: 2, label: "Segment", icon: Cpu },
  { number: 3, label: "Generate", icon: Box },
  { number: 4, label: "Download", icon: Download },
]

export function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <div className="flex items-center justify-center mb-12">
      <div className="flex items-center gap-4 sm:gap-6">
        {steps.map((step, index) => {
          const isCompleted = currentStep > step.number
          const isCurrent = currentStep === step.number
          const Icon = step.icon

          return (
            <div key={step.number} className="flex items-center gap-4 sm:gap-6">
              <div className="flex flex-col items-center gap-2">
                <div
                  className={cn(
                    "flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300",
                    isCompleted && "border-primary bg-primary text-primary-foreground",
                    isCurrent && "border-primary bg-primary/20 text-primary",
                    !isCompleted && !isCurrent && "border-border bg-secondary text-muted-foreground",
                  )}
                >
                  {isCompleted ? <Check className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
                </div>
                <span
                  className={cn(
                    "text-xs font-medium transition-colors",
                    isCurrent ? "text-primary" : "text-muted-foreground",
                  )}
                >
                  {step.label}
                </span>
              </div>

              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div
                  className={cn(
                    "hidden sm:block w-16 lg:w-24 h-0.5 transition-colors duration-300",
                    currentStep > step.number ? "bg-primary" : "bg-border",
                  )}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
