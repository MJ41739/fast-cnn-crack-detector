# Concrete Crack Detection System (Fast CNN)

A production-ready, high-performance deep learning system designed for industrial concrete structure inspection. It features highly optimized CNN architectures, a fast-inference API backend (FastAPI), interactive diagnostics (Grad-CAM thermal heatmap visualization), and an edge-deployment suite (ONNX, INT8 Quantization, Pruning, and TFLite conversion).

---

## Key Features

1. **Neural Architectures**:
   - **Custom Fast CNN**: An optimized, lightweight architecture built from scratch utilizing Depthwise Separable Convolutions to minimize parameters and achieve sub-3ms inference.
   - **MobileNetV2**: Transfer learning configuration with custom classification heads.
   - **EfficientNetB0**: High-accuracy compound-scaled backbone for maximum precision (>98%).
2. **Real-time Grad-CAM Heatmaps**: Computes visual gradient overlays to highlight and explain crack boundaries directly in the interface.
3. **Double-Stage Transfer Training**: Frozen backbone training followed by low-learning-rate fine-tuning to reach target accuracies (>95% target).
4. **Hardware & VRAM Optimization**: Out-of-the-box mixed precision (AMP) support to prevent VRAM OOM on budget GPUs (e.g., 4GB laptop chips).
5. **Real-time Video Feed Scanning**: Live camera video streaming with client-side canvas bounding box overlays and dynamic FPS calculators.
6. **Deployable Edge Formats**: Automated serializations to ONNX, INT8 quantized graph models, and TFLite instructions.

---

## Directory Structure

```
FASTCNN/
│
├── backend/                  # FastAPI Application
│   ├── config.py             # Shared settings schema (Pydantic Settings)
│   ├── main.py               # API endpoints (/predict, /predict-batch, /health)
│   ├── schemas.py            # API request/response validation schemas
│   └── services.py           # ML inference services (PyTorch & ONNX session)
│
├── checkpoints/              # Model weights (.pth) & serialized files (.onnx)
│
├── docker/                   # Deployment containers configs
│   ├── Dockerfile.backend    # API server environment build
│   ├── Dockerfile.frontend   # React asset compiler & Nginx server config
│   └── nginx.conf            # Nginx reverse proxy routes
│
├── frontend/                 # React + Vite application
│   ├── src/
│   │   ├── components/       # Interface tab widgets
│   │   │   ├── DragDropUpload.jsx
│   │   │   ├── BatchUpload.jsx
│   │   │   ├── LiveCamera.jsx
│   │   │   └── PerformanceStats.jsx
│   │   ├── App.jsx           # Main layout frame
│   │   └── index.css         # Styling directives (Tailwind & Glassmorphic CSS)
│   └── package.json
│
├── kaggle_dataset/           # Training concrete scans
│   ├── Positive/             # 20,000 crack images
│   └── Negative/             # 20,000 non-crack images
│
├── models/                   # Deep Learning Models
│   ├── custom_cnn.py         # Custom Fast CNN code
│   ├── mobilenet_v2.py       # MobileNetV2 wrapper
│   ├── efficientnet.py       # EfficientNetB0 wrapper
│   └── factory.py            # Loader factory pattern
│
├── outputs/                  # ROC, Loss curves, Confusion Matrices
│
├── training/                 # Pipeline operations
│   ├── dataset.py            # Dataloaders & Albumentations-style augmentation
│   ├── train.py              # Main training orchestrator (AMP, callbacks)
│   ├── evaluate.py           # Evaluation benchmarks
│   └── export.py             # ONNX and Quantization serializers
│
├── docker-compose.yml        # Orchestration script for containers
├── requirements.txt          # Python dependencies list
└── verify_pipeline.py        # Complete diagnostic test suite
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 to 3.12 (Tested on Python 3.12.0)
- Node.js (v18+) & NPM
- NVIDIA CUDA Toolkit (if using GPU acceleration)

### 1. Python Environment Setup
Clone the repository and install dependencies:
```bash
# Verify python version
python --version

# Install dependencies
pip install -r requirements.txt
```

*Note: PyTorch will automatically configure GPU CUDA acceleration if a matching toolkit is present.*

### 2. Dataset Setup
Download the Kaggle Concrete Crack Images dataset and arrange it under `kaggle_dataset/`:
```
kaggle_dataset/
├── Positive/   # Contains 00001.jpg, 00002.jpg...
└── Negative/   # Contains 00001.jpg, 00002.jpg...
```

---

## How to Run Verification Diagnostics

Before training, run our automated verification pipeline to test your CUDA GPU configurations, PyTorch backpropagation, Grad-CAM modules, and ONNX compiler:
```bash
python verify_pipeline.py
```
If all passes, your environment is ready to start training.

---

## Training and Model Evaluation

The system supports individual training of models as well as full comparison tests.

### 1. Start Model Training
To train the Custom Fast CNN model:
```bash
python training/train.py --model custom_cnn --epochs 15 --batch-size 32
```
To train MobileNetV2 or EfficientNet:
```bash
python training/train.py --model mobilenet_v2
python training/train.py --model efficientnet
```
To train all three models sequentially and compare:
```bash
python training/train.py --model all
```

### 2. Evaluate Models & Generate Reports
Runs inference benchmarks on the 15% test partition, outputting Confusion Matrices and ROC curves inside `outputs/`:
```bash
python training/evaluate.py --model compare
```
This updates `checkpoints/best_model_info.json` pointing the API server to load the overall best-performing model automatically.

### 3. Model Optimization (ONNX & INT8 Quantization)
Convert the best model to ONNX, apply weight pruning, and perform INT8 Dynamic Quantization:
```bash
python training/export.py --model custom_cnn
```

---

## Running the Web System Locally

### 1. Launch FastAPI Backend
```bash
# Run server using Uvicorn
python backend/main.py
```
The server will bind to `http://localhost:8000`. You can inspect raw Swagger documentation at `http://localhost:8000/docs`.

### 2. Launch React Frontend Dashboard
In a separate terminal:
```bash
cd frontend
npm run dev
```
The dashboard will open on `http://localhost:5173`.

---

## Running with Docker (Production Mode)

Docker Compose encapsulates the backend server, builds the React frontend static pages, and coordinates them behind an Nginx reverse-proxy on port 80:

```bash
# Build and run containers in background (use 'docker-compose' if using docker-compose V1)
docker compose up -d --build
```
Access the dashboard at `http://localhost:80` (API routes mapped to `http://localhost:80/api`).

---

## API Documentation

- **`GET /health`**: Returns active model, CUDA GPU parameters, Host RAM, and CPU core diagnostics.
- **`POST /predict?heatmap=true`**: Takes an image file upload. Returns classification prediction, confidence level, raw probability, inference duration, and base64-encoded Grad-CAM overlay heatmap.
- **`POST /predict-batch`**: Accepts list of image files. Executes ultra-fast batched tensor inference (skips Grad-CAM for low-latency target).
- **`POST /change-model`**: Swaps the active inference model in backend memory (e.g. swap from `custom_cnn` to `efficientnet` or `onnx_quantized`).

---

## Edge Quantization & Quantized Models
Our optimization pipeline implements ONNX runtime INT8 weight quantization. The quantized model reduces weight footprint on disk by **~4x** (reducing a 34MB backbone to 8MB) and yields a **~2.5x speed increase** during CPU inferences (ideal for Raspberry Pi, Android, or edge inspect devices).

---

## Future Improvements
- **Edge Deployment**: Implement direct TensorRT compilation pipelines on Jetson Nano inspect modules.
- **Segmentation Models**: Train U-Net or Mask R-CNN architectures to measure physical crack width and depth parameters.
- **Federated Learning**: Integrate remote inspector camera uploads to dynamically fine-tune backbones without aggregating raw image datasets centrally.
