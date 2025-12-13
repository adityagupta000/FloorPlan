
# FloorPlan3D

> AI-powered conversion of 2D floor plans into detailed 3D models using deep learning segmentation

[![Next.js](https://img.shields.io/badge/Next.js-16.0-black?logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com/)

##  UI Snapshort

<div align="center">
  <img src="https://github.com/adityagupta000/FloorPlan/raw/eb384405d190565aa213f420a0426657fc8c44ad/images/1.png" alt="Upload Interface" width="800"/>
  <p><em>Step 1: Upload your 2D floor plan</em></p>
  
  <img src="https://github.com/adityagupta000/FloorPlan/raw/eb384405d190565aa213f420a0426657fc8c44ad/images/2.png" alt="AI Segmentation" width="800"/>
  <p><em>Step 2: AI-powered semantic segmentation</em></p>
  
  <img src="https://github.com/adityagupta000/FloorPlan/raw/eb384405d190565aa213f420a0426657fc8c44ad/images/3.png" alt="3D Generation" width="800"/>
  <p><em>Step 3: Generate 3D model from segmented mask</em></p>
  
  <img src="https://github.com/adityagupta000/FloorPlan/raw/eb384405d190565aa213f420a0426657fc8c44ad/images/4.png" alt="3D Preview" width="800"/>
  <p><em>Step 4: Real-time 3D preview with orbit controls</em></p>
  
  <img src="https://github.com/adityagupta000/FloorPlan/raw/eb384405d190565aa213f420a0426657fc8c44ad/images/5.png" alt="Download & Export" width="800"/>
  <p><em>Step 5: Download OBJ file for use in Blender, SketchUp, etc.</em></p>
</div>

##  Overview

FloorPlan3D is an end-to-end system that transforms 2D architectural floor plans into high-quality 3D models. Using a custom ResNet50-UNet architecture with attention gates, the system performs semantic segmentation to identify walls, doors, windows, and floors, then generates industry-standard OBJ files with proper geometry and vertex colors.

### Key Features

- **AI-Powered Segmentation**: ResNet50-UNet with attention mechanisms for precise detection of thin features
- **5-Class Semantic Segmentation**: Background, walls, doors, windows, and floors
- **Test-Time Augmentation**: Rotation-invariant predictions (0°, 90°, 180°, 270°)
- **3D Model Generation**: Exports to Wavefront OBJ format with vertex colors
- **Interactive Preview**: Real-time 3D visualization using Three.js
- **Modern Web Interface**: Built with Next.js 16, TypeScript, and Tailwind CSS

##  Architecture

### Frontend (Next.js + TypeScript)
- **Framework**: Next.js 16.0 with App Router
- **UI Components**: Shadcn/ui with Radix UI primitives
- **3D Rendering**: React Three Fiber + Three.js (r128)
- **Styling**: Tailwind CSS 4.1 with custom design system
- **Features**: Dark mode, responsive design, drag-and-drop uploads

### Backend (Python + Flask)
- **Framework**: Flask 3.0 with CORS support
- **Deep Learning**: PyTorch 2.1 with CUDA acceleration
- **Model**: ResNet50-UNet with Attention Gates
- **3D Processing**: Trimesh, Shapely for geometry manipulation
- **Image Processing**: OpenCV, Pillow

### Model Architecture
- **Encoder**: ResNet50 (ImageNet pretrained)
- **Decoder**: U-Net with skip connections
- **Attention Gates**: 4 levels for fine detail preservation
- **Input Resolution**: 512×512
- **Output Classes**: 5 (background, walls, doors, windows, floors)

##  Getting Started

### Prerequisites

- **Node.js**: 18.x or higher
- **Python**: 3.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **Git**: For cloning the repository

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/adityagupta000/floorPlan.git
cd floorPlan
```

#### 2. Frontend Setup

```bash
# Install dependencies
npm install

# Set up environment variables
echo "NEXT_PUBLIC_API_URL=http://localhost:5000/api" > .env.local

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

#### 3. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your trained model
# Copy best_model.pth to backend/ directory

# Start Flask server
python app.py
```

The backend API will be available at `http://localhost:5000`

### Project Structure

```
floorplan3d/
├── app/                          # Next.js app directory
│   ├── layout.tsx               # Root layout with metadata
│   └── page.tsx                 # Main page component
├── components/                   # React components
│   ├── workflow/                # Step-by-step workflow
│   │   ├── upload-step.tsx
│   │   ├── segment-step.tsx
│   │   ├── generate-step.tsx
│   │   └── download-step.tsx
│   ├── model-3d-viewer.tsx      # Three.js 3D viewer
│   ├── processing-workflow.tsx   # Main workflow orchestrator
│   └── ui/                      # Shadcn UI components
├── backend/                      # Python Flask backend
│   ├── app.py                   # Flask REST API
│   ├── inference.py             # Model inference with TTA
│   ├── model.py                 # ResNet50-UNet architecture
│   ├── obj_generator.py         # 3D model generation
│   ├── requirements.txt         # Python dependencies
│   └── best_model.pth          # Trained model weights
├── uploads/                      # Temporary uploaded images
├── outputs/                      # Generated masks and OBJ files
└── package.json                 # Node.js dependencies
```

##  Usage

### Web Interface

1. **Upload**: Drag and drop or click to upload a 2D floor plan (PNG, JPG, JPEG, max 16MB)
2. **Segment**: AI analyzes the image and identifies architectural elements
3. **Generate**: Creates a 3D OBJ model with proper geometry and colors
4. **Download**: Preview in real-time and download the OBJ file

### API Endpoints

#### Health Check
```bash
GET /api/health
```

#### Upload Image
```bash
POST /api/upload
Content-Type: multipart/form-data
Body: { image: File }
```

#### Run Segmentation
```bash
POST /api/segment
Content-Type: application/json
Body: { filename: string }
```

#### Generate 3D Model
```bash
POST /api/generate3d
Content-Type: application/json
Body: { mask_filename: string }
```

#### Download File
```bash
GET /api/download/:filename
```

##  Model Training

### Dataset Preparation

Organize your dataset as follows:

```
dataset/
├── ImagesReal/
│   ├── train/
│   ├── val/
│   └── test/
└── MasksReal/
    ├── train/
    ├── val/
    └── test/
```

Mask pixel values: 0=background, 1=walls, 2=doors, 3=windows, 4=floors

### Training Script

```python
python train_old.py
```

**Training Features:**
- Focal Loss for class imbalance
- Boundary-aware loss for thin features
- Small object loss for windows
- Test-time augmentation (rotation 0-360°)
- Class weights: [0.5, 3.0, 3.5, 5.0, 1.0]
- AdamW optimizer with ReduceLROnPlateau scheduler
- Early stopping with patience=10

**Hyperparameters:**
- Batch size: 4
- Learning rate: 1e-4
- Epochs: 50 (with early stopping)
- Input size: 512×512
- Weight decay: 1e-4

## 📊 Model Performance

- **Dice Coefficient**: Measures segmentation accuracy
- **IoU (Intersection over Union)**: Measures overlap quality
- **Loss**: Combined CrossEntropy + Dice + Boundary + Small Object Loss

The model uses Test-Time Augmentation (TTA) with 4 rotations for rotation-invariant predictions.

##  3D Model Output

**Geometry Details:**
- **Walls**: 0.8m height with proper openings
- **Doors**: Full height frames at floor level
- **Windows**: Frames at realistic height (0.4m base, 0.3m height)
- **Floor**: 2cm thick slab with tile pattern
- **Scale**: 1 pixel = 1cm

**Colors (Vertex Colors):**
- Walls: Gray (#B4B4B4)
- Doors: Brown (#A5372D)
- Window Frames: Amber (#FFB41E)
- Glass: Blue (#32AAFF, transparent)
- Floor: Beige (#D2BE96)

**Compatible Software:**
- Blender
- SketchUp
- AutoCAD
- 3ds Max
- Any OBJ-compatible 3D software

##  Configuration

### Backend Configuration (`backend/obj_generator.py`)

```python
HEIGHT_SCALE = 1.5          # Overall height multiplier
WALL_HEIGHT = 0.8           # Wall height in meters
WINDOW_BASE_HEIGHT = 0.4    # Window sill height
WINDOW_HEIGHT = 0.3         # Window opening height
DOOR_HEIGHT = 0.8           # Door height
TILE_SIZE = 0.3             # Floor tile size
```

### Frontend Configuration

```typescript
// Environment variable
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

##  Troubleshooting

### Backend Issues

**CUDA Out of Memory**
- Reduce batch size in `train_old.py`
- Use CPU inference: Set `DEVICE = torch.device("cpu")`

**Model Not Found**
- Ensure `best_model.pth` is in `backend/` directory
- Check file permissions

**Import Errors**
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Frontend Issues

**3D Viewer Not Loading**
- Check browser console for errors
- Ensure Three.js dependencies are installed: `npm install three @react-three/fiber @react-three/drei`

**API Connection Failed**
- Verify backend is running on port 5000
- Check CORS configuration in `backend/app.py`

##  Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

##  Acknowledgments

- ResNet50 architecture from torchvision
- U-Net concept from Ronneberger et al.
- Attention gates inspired by Oktay et al.
- Shadcn/ui component library
- React Three Fiber community

---

**Built with ❤️ using Next.js, PyTorch, and Three.js**
