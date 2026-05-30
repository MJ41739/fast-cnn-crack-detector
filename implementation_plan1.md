# Implementation Plan - Production-Ready Crack Detection System

This document outlines the design, architecture, and implementation steps for building a complete, production-ready concrete crack detection system. The system will feature optimized deep learning model training, a FastAPI backend with Grad-CAM heatmaps, a React+Vite frontend with batch upload and webcam support, and edge deployment features like ONNX export, model quantization, and TFLite conversion.

---

## User Review Required

> [!IMPORTANT]
> **1. Deep Learning Framework Choice**
> We recommend **PyTorch** over TensorFlow/Keras for this project. PyTorch 2.2+ fully supports Python 3.12 (the version installed on your machine) and provides highly modular API structures for:
> - Post-Training Quantization (PTQ) and model pruning.
> - Direct and robust ONNX export, which can then be converted to TFLite or TensorRT.
> - Custom Grad-CAM implementation without class/layer naming complexities common in newer Keras 3.x setups.
> 
> *If you strictly prefer TensorFlow/Keras, please let us know; otherwise, the plan proceeds with PyTorch.*

> [!WARNING]
> **2. GPU VRAM Limitation**
> Your system has an NVIDIA RTX 3050 Laptop GPU with **4GB of VRAM**.
> Since we need to train three models (Custom Fast CNN, MobileNetV2, and EfficientNetB0) on 40,000 images, memory management is critical. We will implement:
> - **Mixed Precision Training (AMP)** to reduce VRAM footprint.
> - Configurable batch size (default to `32` or `64`) to prevent Out Of Memory (OOM) errors.
> - Pre-scaled image dimensions (224x224) which is standard for pretrained backbones and efficient for custom training.

---

## Open Questions

1. **Dataset Split Ratio**: The plan uses a standard split of 70% training, 15% validation, and 15% testing. Do you have any specific split preferences?
2. **Transfer Learning Strategy**: For MobileNetV2 and EfficientNetB0, should we freeze the backbone entirely and only train the classifier, or perform two-stage training (initial head training followed by fine-tuning at a very low learning rate)? *We propose two-stage training for maximum accuracy (>95% target).*

---

## Proposed Changes

We will construct a clean, modular structure. Below is the proposed file organization under `c:\Users\Mayur Jadhav\OneDrive\Desktop\FASTCNN`.

```
c:\Users\Mayur Jadhav\OneDrive\Desktop\FASTCNN\
├── kaggle_dataset/      # Pre-existing dataset (Negative/ and Positive/ directories)
├── models/              # Model architectures
│   ├── __init__.py
│   ├── custom_cnn.py    # Custom Fast CNN architecture
│   ├── mobilenet_v2.py  # MobileNetV2 transfer learning model
│   ├── efficientnet.py  # EfficientNetB0 transfer learning model
│   └── factory.py       # Factory pattern to load/build models
├── backend/             # FastAPI backend application
│   ├── __init__.py
│   ├── main.py          # FastAPI application routes
│   ├── config.py        # System configuration settings (Pydantic / Config parser)
│   ├── schemas.py       # Pydantic schemas for requests/responses
│   └── services.py      # Core inference and Grad-CAM generation service
├── frontend/            # React + Vite frontend
│   ├── public/          # Static assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── DragDropUpload.jsx   # Single image upload and prediction display
│   │   │   ├── BatchUpload.jsx      # Batch prediction grid
│   │   │   ├── LiveCamera.jsx       # Real-time webcam crack detection with FPS
│   │   │   └── PerformanceStats.jsx # UI for viewing model statistics
│   │   ├── App.jsx                  # Main interface logic and layout
│   │   ├── index.css                # Tailwind import and custom CSS styling
│   │   └── main.jsx
│   ├── package.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── vite.config.js
├── training/            # Training and evaluation pipeline
│   ├── __init__.py
│   ├── dataset.py       # PyTorch Dataset and augmentations (Albumentations/Torchvision)
│   ├── train.py         # Model training script with AMP, early stopping, and checkpoints
│   ├── evaluate.py      # Evaluation script calculating ROC-AUC, Confusion Matrix, etc.
│   └── export.py        # Export to ONNX, TFLite conversion, pruning, and quantization
├── utils/               # Shared utilities
│   ├── __init__.py
│   ├── gradcam.py       # PyTorch Grad-CAM utility for heatmap visualization
│   └── visualizer.py    # Functions to plot training curves, confusion matrix, ROC curve
├── outputs/             # Generated evaluation plots, confusion matrices, ROC curves
├── checkpoints/         # Model weights (.pth) and exported models (.onnx, .tflite)
├── docker/              # Docker configuration files
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── requirements.txt     # Python requirements
├── README.md            # Comprehensive project documentation
└── docker-compose.yml   # Multi-container orchestrator
```

### Component Details

#### 1. Models & Architecture (`models/`)
- **[NEW] [custom_cnn.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/models/custom_cnn.py)**: Lightweight Custom CNN using Depthwise Separable Convolutions (to minimize parameters and latency), Batch Normalization, ReLU activation, MaxPooling, and Dropout. Ends with a global average pooling layer and dense projection.
- **[NEW] [mobilenet_v2.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/models/mobilenet_v2.py)**: PyTorch MobileNetV2 with custom linear classifier head.
- **[NEW] [efficientnet.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/models/efficientnet.py)**: PyTorch EfficientNetB0 with custom linear classifier head.
- **[NEW] [factory.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/models/factory.py)**: Simple factory function to load models by name, handles input channel mapping and output head configuration.

#### 2. Training Pipeline (`training/`)
- **[NEW] [dataset.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/training/dataset.py)**:
  - Custom dataset with automatic Train/Val/Test partitioning of folders.
  - Image resizing to 224x224 and normalization.
  - Clean corruption handling: uses Try/Except block inside image loading; if an image is corrupted, it prints a warning and falls back to a different index or a blank tensor.
  - Robust augmentation using Albumentations or torchvision: random rotation, horizontal/vertical flips, zoom/random crop, brightness/contrast adjustments, and Gaussian noise injection.
- **[NEW] [train.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/training/train.py)**:
  - Automatically detects CUDA GPU.
  - Mixed precision training using `torch.cuda.amp.GradScaler` and `torch.autocast`.
  - Implements checkpoints saving the current epoch state and the best model weights.
  - Custom early stopping callback based on validation loss/accuracy.
  - Learning rate scheduling via `ReduceLROnPlateau`.
  - Supports resuming training by loading checkpoint files.
- **[NEW] [evaluate.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/training/evaluate.py)**:
  - Evaluates models on the test split.
  - Computes Accuracy, F1-score, Precision, Recall, Confusion Matrix, and ROC-AUC.
  - Saves matplotlib figures of the Confusion Matrix and ROC Curve to the `outputs/` folder.
- **[NEW] [export.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/training/export.py)**:
  - Exports PyTorch models to ONNX formats.
  - Supports model quantization (PTQ to INT8) to optimize inference on CPUs.
  - Supports model pruning (structured/unstructured weight pruning) to reduce model memory footprints.
  - Handles exporting to TFLite (via a converter script or a checkpoint representation).

#### 3. Utilities (`utils/`)
- **[NEW] [gradcam.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/utils/gradcam.py)**:
  - Custom PyTorch Grad-CAM implementation. Extracts feature maps from the final convolutional layer of the model, calculates the gradients of the target class, weights the channels, generates a heatmap overlay, and merges it with the original input image.
- **[NEW] [visualizer.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/utils/visualizer.py)**:
  - Utility to plot training accuracy/loss curves and save them.

#### 4. Backend Server (`backend/`)
- **[NEW] [config.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/backend/config.py)**: Configuration loader using `pydantic-settings` to parse environments.
- **[NEW] [main.py](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/backend/main.py)**: FastAPI web app implementing CORS middleware.
  - `/predict`: Accepts an uploaded image file, performs preprocessing, runs inference (using the chosen best model), runs Grad-CAM, and returns a JSON payload containing prediction (`crack` or `no-crack`), probability, confidence, and a base64-encoded Grad-CAM overlay image.
  - `/predict-batch`: Accepts a list of uploaded images, performs fast batched predictions, and returns a JSON array of predictions.
  - `/health`: Performs health checks and returns current memory usage, uptime, selected active model name, and CUDA availability status.
  - `/change-model`: API to hot-swap the selected model (Custom, MobileNetV2, EfficientNetB0, ONNX, Quantized) in real-time.

#### 5. Frontend UI (`frontend/`)
- **[NEW] [package.json](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/frontend/package.json)**, **[vite.config.js](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/frontend/vite.config.js)**: Configures a React + Vite application.
- **[NEW] [tailwind.config.js](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/frontend/tailwind.config.js)**, **[postcss.config.js](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/frontend/postcss.config.js)**: Configures Tailwind CSS styling.
- **[NEW] [index.css](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/frontend/src/index.css)**: Implements custom styles, glassmorphism utilities, and modern color palettes.
- **[NEW] [App.jsx](file:///c:/Users/Mayur/Jadhav/OneDrive/Desktop/FASTCNN/frontend/src/App.jsx)**: Implements tabbed navigation:
  - **Single Prediction Tab**: Drag-and-drop upload container, prediction results layout, side-by-side comparison of original image vs. Grad-CAM heatmap, confidence indicator bar.
  - **Batch Prediction Tab**: File selector allowing multiple image uploads. Displays a grid layout of thumbnails with their crack prediction, confidence, processing time, and download options for reports.
  - **Live Camera Tab**: Stream from local webcam, feed frames to the FastAPI backend, show bounding overlay with real-time FPS counter.
  - **Model Performance Tab**: Displays model metrics (Accuracy, F1-score, Latency) comparing Custom Fast CNN, MobileNetV2, and EfficientNetB0, alongside saved Confusion Matrices and ROC Curves.

---

## Verification Plan

### Automated Verification
1. **Unit and Pipeline Tests**:
   - Write a lightweight verify script `verify_pipeline.py` in `training/` that creates dummy tensors, runs the data loaders, trains the models for 1 epoch, and tests model saving/loading.
   - Run verification checks on model exports (PyTorch -> ONNX).
2. **API Endpoint Testing**:
   - Run backend server in testing mode using `pytest` or a fast curl-based verify script to test `/health`, `/predict` (with single test image), and `/predict-batch`.
3. **Inference Latency Test**:
   - Run a benchmark script tracking inference time for each model (PyTorch, ONNX, Quantized PyTorch) to confirm inference latency remains < 100ms on both CPU and GPU.

### Manual Verification
1. **Web App Evaluation**:
   - Deploy backend and frontend locally.
   - Upload crack and non-crack images manually via the drag-and-drop area.
   - Inspect the Grad-CAM heatmap overlays to check if the model correctly localizes crack boundaries.
   - Test webcam live detection to check frame rate (FPS) and prediction stability.
