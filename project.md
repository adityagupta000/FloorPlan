# 2D Blueprint to 3D Model Converter

An automated system for converting 2D architectural blueprints into accurate 3D models using deep learning and computer vision.

---

## Abstract

This project presents an automated system for converting 2D architectural blueprints into accurate 3D models using deep learning and computer vision. A ResNet50-based Attention U-Net semantic segmentation model is developed to detect structural elements such as walls, doors, windows, and floor regions. A custom SVG→XML→Mask data-generation pipeline is used to create high-quality pixel-wise training masks from blueprint vector files. The predicted segmentation mask is further processed into geometric polygons and extruded to form a structured 3D model using a custom OBJ-generation module. The 3D output is rendered interactively on a web-based viewer built with Three.js, enabling smooth rotation, zooming, and inspection of the generated structure. The system is designed to operate reliably on digital, scanned, and hand-drawn floor plans with offline support, providing a practical tool for civil, architectural, and tactical applications.

---

## Problem Statement

Manual interpretation of architectural blueprints and the creation of corresponding 3D models is time-consuming, skill-dependent, and prone to human error. Traditional computer-vision methods fail to consistently detect thin structural elements such as walls, windows, and doors across diverse blueprint styles. A fully automated and robust system is needed to extract architectural semantics from 2D layouts and convert them into accurate 3D models, while ensuring adaptability to various drawing formats including scanned images and hand-drawn plans, and supporting offline deployment for secure usage scenarios.

---

## Objectives

- To develop an automated system that converts 2D architectural blueprints into accurate 3D models using deep learning and computer vision techniques.

- To automatically detect and interpret key structural components—such as walls, doors, windows, and floor regions—through segmentation and image analysis, thereby reducing manual effort and ensuring consistent results.

- To ensure reliable system performance across diverse blueprint types, including digital, scanned, annotated, and hand-drawn layouts, while supporting offline processing for secure and remote usage where internet access may be limited.

---

## System Architecture

The system consists of five major layers:

### 1. Input Layer

Users upload a 2D floor plan in PNG, JPG, or SVG format.

### 2. Preprocessing Layer

- SVG files are converted to XML polygons
- XML polygons are rasterized into pixel-level segmentation masks
- Images are resized and normalized for model input

### 3. Deep Learning Segmentation Layer

- ResNet50 extracts high-level features from the blueprint
- Attention U-Net decodes the features into a full-resolution mask
- Output is a 5-class semantic mask (wall, door, window, floor, background)

### 4. 3D Reconstruction Layer

- Mask → polygon extraction
- Wall boundaries are traced
- Walls are extruded into 3D height
- Floor mesh is created
- Openings are carved out for windows and doors
- Final model exported as an OBJ file

### 5. Visualization Layer

- OBJ file displayed in browser using Three.js
- Supports rotate, zoom, pan, and mode switching (Civil/Tactical)
- Works offline with auto-sync capability

---

## Methodology

### A) Dataset Generation (SVG→XML→Mask)

Since floor plans do not come with masks, a custom pipeline was built:

**1. SVG to XML Conversion**

- Extract polygons representing walls, doors, windows, and room spaces
- Normalize class labels (e.g., SpaceKitchen → Floor)
- Store polygon coordinates with six-decimal precision

**2. XML to Mask Conversion**

- Render polygons into 512×512 masks
- Floor rendered first, followed by walls, doors, and windows
- Walls drawn with controlled thickness
- Small objects (<25 pixels) are removed for better training stability

**3. Dataset Splitting**

- Images and masks split into train/validation/test groups

### B) Deep Learning Model (ResNet50 + Attention U-Net)

- **ResNet50 Encoder:** Learns complex structural patterns
- **U-Net Decoder:** Reconstructs pixel-wise predictions
- **Attention Gates:** Focus on thin architectural elements
- **Output:** 5-channel semantic mask

### C) Training Strategy

- Training performed for 60 epochs
- **Loss functions:**
  - Focal Loss (handles class imbalance)
  - Boundary Loss (sharpens edges)
  - Small-Object Loss (improves door/window accuracy)
  - Dice Loss (overall segmentation accuracy)
- **Augmentations:** rotations, flips, color jitter
- **Optimizer:** Adam

### D) Inference & Post-Processing

- Blueprint is normalized and passed through the model
- Output mask is converted to polygons
- Polygons used to construct a 3D mesh
- OBJ model is generated and stored

### E) 3D Visualization

- Three.js loads and renders OBJ
- User can explore model interactively
- Supports Civil and Tactical modes

---

## Algorithms Used

### Deep Learning Algorithms

- ResNet50 feature extractor
- Attention U-Net
- Focal Loss, Boundary Loss, Small-Object Loss
- Dice Score, IoU metrics

### Mask Generation Algorithms

- Polygon filling
- Wall thickness rendering
- Small-object removal
- Class-priority rendering (Floor → Wall → Door/Window)

### 3D Reconstruction Algorithms

- Polygon boundary tracing
- Mesh extrusion
- OBJ vertex/face generation

### 3D Rendering Algorithms

- Three.js WebGL rendering
- OBJ loader
- Orbit controls (zoom, rotate, pan)

---

## Workflow

1. User uploads blueprint (PNG/JPG/SVG)
2. SVG→XML→Mask conversion creates pixel-wise labels
3. Deep learning model predicts semantic mask
4. Mask boundaries converted into geometric polygons
5. Polygons extruded into 3D structure (OBJ model)
6. Viewer loads OBJ for interactive visualization
7. User switches between Civil and Tactical modes
8. Offline processing supported with sync-on-reconnect

---

## Results

- Achieved **94.51% segmentation accuracy**
- Clear wall boundary detection
- Improved door and window segmentation through small-object loss
- Successful conversion of mask into a structured 3D model
- Efficient real-time rendering using Three.js
- Works on multiple blueprint styles including scanned and hand-drawn layouts

---

## Conclusion

The project successfully automates the conversion of 2D architectural floor plans into interactive 3D models. By combining a custom SVG→XML→Mask preprocessing pipeline with a ResNet50-based Attention U-Net, the system achieves high segmentation accuracy and supports diverse blueprint styles. The 3D reconstruction module accurately builds structural models, and the Three.js viewer provides an intuitive interface for civil and tactical applications. The system is reliable, offline-capable, and significantly reduces manual modeling time.

---

## Technologies Used

- **Deep Learning:** ResNet50, Attention U-Net, PyTorch/TensorFlow
- **Computer Vision:** OpenCV, Polygon extraction
- **3D Processing:** Custom OBJ generation
- **Visualization:** Three.js, WebGL
- **File Formats:** SVG, XML, PNG, JPG, OBJ

---

## Future Enhancements

- Support for multi-story buildings
- Real-time collaboration features
- Advanced material and texture mapping
- Integration with BIM (Building Information Modeling) systems
- Mobile application support
