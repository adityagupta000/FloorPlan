import { Box } from "lucide-react"
import Link from "next/link"

export function Footer() {
  return (
    <footer className="border-t border-border bg-card/50 py-12">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Box className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="text-lg font-semibold">FloorPlan3D</span>
          </div>

          {/* Pipeline Info */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground font-mono">
              Image Upload → AI Segmentation (ResNet50-UNet) → 3D Conversion (Trimesh) → OBJ Export
            </p>
          </div>

          {/* Links */}
          <div className="flex items-center gap-6">
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Documentation
            </Link>
            <Link href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              GitHub
            </Link>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border text-center">
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} FloorPlan3D. AI-powered floor plan to 3D model conversion.
          </p>
        </div>
      </div>
    </footer>
  )
}
