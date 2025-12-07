import { Cpu, Layers, Download, Zap, Shield, Code2 } from "lucide-react"

const features = [
  {
    icon: Cpu,
    title: "ResNet50-UNet Architecture",
    description:
      "State-of-the-art deep learning model with attention gates for precise segmentation of thin features like walls.",
  },
  {
    icon: Layers,
    title: "5-Class Segmentation",
    description: "Automatically identifies background, walls, doors, windows, and floors with high accuracy.",
  },
  {
    icon: Zap,
    title: "Fast Processing",
    description: "GPU-accelerated inference delivers results in seconds, not minutes. Optimized for production use.",
  },
  {
    icon: Download,
    title: "Standard OBJ Export",
    description: "Export to industry-standard Wavefront OBJ format compatible with all major 3D software.",
  },
  {
    icon: Shield,
    title: "Privacy First",
    description: "Your floor plans are processed locally on your server. No data is sent to external services.",
  },
  {
    icon: Code2,
    title: "REST API",
    description: "Simple REST API for easy integration into your existing workflows and applications.",
  },
]

export function Features() {
  return (
    <section id="features" className="py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">Powerful Features</h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Built with cutting-edge AI technology for architects, designers, and developers
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group p-6 rounded-2xl border border-border bg-card hover:border-primary/50 transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Technical Stack */}
        {/* <div className="mt-20 p-8 rounded-2xl border border-border bg-card">
          <h3 className="text-xl font-semibold mb-6 text-center">Technical Stack</h3>
          <div className="flex flex-wrap justify-center gap-4">
            {["PyTorch", "Flask", "OpenCV", "Trimesh", "Shapely", "NumPy", "Next.js", "Three.js"].map((tech) => (
              <span key={tech} className="px-4 py-2 rounded-full bg-secondary text-sm font-medium">
                {tech}
              </span>
            ))}
          </div>
        </div> */}
      </div>
    </section>
  )
}
