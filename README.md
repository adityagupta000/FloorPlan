# FloorPlan - Object Detection Application

Full-stack ML application with Next.js frontend and Python backend for floor plan object detection.

## ðŸ“ Project Structure

\\\
FloorPlan/
â”œâ”€â”€ backend/          # Python API for model inference
â”‚   â”œâ”€â”€ app.py
â”‚   â”œâ”€â”€ model.py
â”‚   â”œâ”€â”€ inference.py
â”‚   â””â”€â”€ requirements.txt
â”œâ”€â”€ frontend/         # Next.js React application
â”‚   â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ components/
â”‚   â”œâ”€â”€ lib/
â”‚   â””â”€â”€ public/
â””â”€â”€ training/         # Training scripts and sample data
    â”œâ”€â”€ ImagesReal/   # 10 sample images
    â””â”€â”€ masksReal/    # 10 sample masks
\\\

## ðŸš€ Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- pnpm (or npm/yarn)

### Backend Setup

\\\ash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model file (838 MB)
# Download from: [ADD GOOGLE DRIVE LINK HERE]
# Place best_model.pth in backend/ directory

# Run backend
python app.py
\\\

### Frontend Setup

\\\ash
cd frontend

# Install dependencies
pnpm install

# Run development server
pnpm dev
\\\

Open [http://localhost:3000](http://localhost:3000) in your browser.

## ðŸ“¦ Model File

The trained model file (\est_model.pth\ - 838 MB) is too large for GitHub.

**Download:** [ADD GOOGLE DRIVE LINK HERE]

Place it in the \ackend/\ directory before running the application.

## âœ¨ Features

- Real-time object detection on floor plans
- Interactive web interface
- REST API for predictions
- Sample dataset included for testing
- Responsive design

## ðŸ› ï¸ Tech Stack

- **Frontend:** Next.js 14, React, Tailwind CSS, TypeScript
- **Backend:** Python, FastAPI/Flask, PyTorch
- **ML:** PyTorch, Computer Vision, Object Detection

## ðŸ“ API Endpoints

- \POST /predict\ - Upload image for object detection
- \GET /health\ - Health check endpoint

## ðŸ¤ Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ðŸ“„ License

[Add your license here]

## ðŸ‘¤ Author

Aditya Gupta
- GitHub: [@adityagupta000](https://github.com/adityagupta000)
