import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Setup slowapi for rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import settings
from backend.schemas import (
    HealthResponse, PredictionResponse, 
    BatchPredictionResponse, ChangeModelRequest, ChangeModelResponse
)
from backend.services import inference_service

# Load optional psutil for system health stats
try:
    import psutil
except ImportError:
    psutil = None

import torch

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for high-performance Concrete Crack Detection.",
    version="1.0.0"
)

# Attach limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, Vercel proxy rewrite handles this securely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Gathers system health and hardware diagnostics details."""
    cpu_cores = os.cpu_count() or 0
    
    # Read RAM statistics
    mem_usage = 0.0
    if psutil:
        mem = psutil.virtual_memory()
        mem_usage = mem.percent
    
    # Read GPU / CUDA statistics
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
    vram_mb = 0.0
    if cuda_avail:
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
        
    return HealthResponse(
        status="Healthy",
        app_name=settings.APP_NAME,
        active_model=inference_service.active_model_name,
        device=str(inference_service.device),
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        vram_allocated_mb=vram_mb,
        cpu_cores=cpu_cores,
        memory_usage_percent=mem_usage
    )

@app.get("/model-info", tags=["Diagnostics"])
def get_model_info():
    """Returns details about the active deep learning model and hardware configuration."""
    is_onnx = inference_service.onnx_session is not None
    model_type = "ONNX Runtime" if is_onnx else "PyTorch"
    
    return {
        "active_model": inference_service.active_model_name,
        "device": str(inference_service.device),
        "model_type": model_type,
        "onnx_optimized": is_onnx,
        "quantized": "quantized" in inference_service.active_model_name,
        "image_size": f"{settings.IMAGE_SIZE}x{settings.IMAGE_SIZE}",
        "precision": "int8" if "quantized" in inference_service.active_model_name else "float32"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("100/minute")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    heatmap: bool = Query(True, description="Generate Grad-CAM heatmap visualization overlay")
):
    """Run prediction on a single concrete image, returning class and Grad-CAM base64 heatmap."""
    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: .jpg, .jpeg, .png, .bmp"
        )
        
    # Validate MIME type
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type '{file.content_type}'. Must be image/jpeg, image/png, or image/bmp."
        )

    # Check file size (Max 5MB)
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)  # Reset seek pointer

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the 5MB limit. Uploaded file is {file_size / (1024 * 1024):.2f}MB."
        )
        
    try:
        # Read raw image bytes
        contents = await file.read()
        
        # Run prediction
        result = inference_service.predict(contents, file.filename, generate_heatmap=heatmap)
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
@limiter.limit("100/minute")
async def predict_batch_images(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """Run predictions on multiple uploaded concrete images using optimized batching."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded."
        )
        
    images_list = []
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            if file.content_type in ["image/jpeg", "image/png", "image/bmp"]:
                # Check file size (Max 5MB)
                file.file.seek(0, os.SEEK_END)
                file_size = file.file.tell()
                file.file.seek(0)
                
                if file_size <= 5 * 1024 * 1024:
                    try:
                        contents = await file.read()
                        images_list.append((contents, file.filename))
                      
                    except Exception:
                        # Add placeholder prediction if file read fails
                        pass
                
    if not images_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid image files found in upload list. Allowed: .jpg, .jpeg, .png, .bmp under 5MB each."
        )
        
    try:
        # Run batched predictions
        result = inference_service.predict_batch(images_list)
        return BatchPredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )

@app.post("/change-model", response_model=ChangeModelResponse, tags=["Diagnostics"])
def change_inference_model(payload: ChangeModelRequest):
    """Hot-swap the active deep learning model in production (supports PyTorch & ONNX models)."""
    try:
        inference_service.load_model(payload.model_name)
        return ChangeModelResponse(
            success=True,
            message=f"Model successfully switched to '{payload.model_name}'",
            active_model=inference_service.active_model_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load model '{payload.model_name}': {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Start uvicorn server on port 8000 when run directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
