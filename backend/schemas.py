from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    app_name: str
    active_model: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    vram_allocated_mb: float
    cpu_cores: int
    memory_usage_percent: float

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    probability: float
    heatmap_image: Optional[str] = None  # Base64 data URL
    inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    total_images: int
    predictions: List[PredictionResponse]
    average_latency_ms: float

class ChangeModelRequest(BaseModel):
    model_name: str  # 'custom_cnn', 'mobilenet_v2', 'efficientnet', 'onnx', 'onnx_quantized'

class ChangeModelResponse(BaseModel):
    success: bool
    message: str
    active_model: str
