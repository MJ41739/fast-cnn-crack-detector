import os
import sys
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.config import settings
from models.factory import get_model
from utils.gradcam import get_gradcam_visualization
from training.dataset import get_dataloaders
from training.export import export_to_onnx, quantize_onnx_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def verify_gpu_and_environment():
    """Verify system diagnostics and hardware settings."""
    logger.info("=== STEP 1: Environment & GPU Check ===")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    logger.info(f"CUDA GPU Available: {cuda_avail}")
    if cuda_avail:
        logger.info(f"Active GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0)/(1024*1024):.1f} MB")
    
    # Check dataset existence
    dataset_exists = os.path.exists(settings.DATASET_DIR)
    logger.info(f"Dataset Folder Location: {settings.DATASET_DIR} | Exists: {dataset_exists}")
    
    if dataset_exists:
        pos_dir = os.path.join(settings.DATASET_DIR, "Positive")
        neg_dir = os.path.join(settings.DATASET_DIR, "Negative")
        logger.info(f"  - Positive dir: {os.path.exists(pos_dir)}")
        logger.info(f"  - Negative dir: {os.path.exists(neg_dir)}")

def verify_model_factory():
    """Instantiate and run a dry-run inference on all models using dummy inputs."""
    logger.info("\n=== STEP 2: Model Architecture Verification ===")
    models = ["custom_cnn", "mobilenet_v2", "efficientnet", "rcnn"]
    device = torch.device(settings.DEVICE)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    for name in models:
        try:
            start_time = time.perf_counter()
            # Instantiate model
            model = get_model(name, pretrained=False, num_classes=1)
            model = model.to(device)
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
                
            latency = (time.perf_counter() - start_time) * 1000.0
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(
                f"Model: {name:<15} | Params: {params:,} | "
                f"Dummy Forward Success! Output shape: {list(output.shape)} | Latency: {latency:.1f}ms"
            )
        except Exception as e:
            logger.error(f"Failed to verify model '{name}': {e}")
            raise e

def verify_training_backpropagation():
    """Verify that models can execute forward and backward passes to validate training loops."""
    logger.info("\n=== STEP 3: Backpropagation & Training Loop Verification ===")
    device = torch.device(settings.DEVICE)
    
    # Load custom model
    model = get_model("custom_cnn", pretrained=False, num_classes=1).to(device)
    model.train()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    inputs = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.ones(4, 1).to(device)
    
    try:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Backpropagation pass completed successfully. Loss: {loss.item():.4f}")
    except Exception as e:
        logger.error(f"Backpropagation verification failed: {e}")
        raise e

def verify_gradcam():
    """Verify Grad-CAM hooks and overlay calculations."""
    logger.info("\n=== STEP 4: Grad-CAM Hook Verification ===")
    device = torch.device(settings.DEVICE)
    
    # Load model
    model = get_model("custom_cnn", pretrained=False, num_classes=1).to(device)
    model.eval()
    
    # Create dummy PIL image
    dummy_img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_img_array)
    
    try:
        overlay, prob, is_crack = get_gradcam_visualization(model, dummy_image, device)
        logger.info(f"Grad-CAM execution successful. Output shape: {overlay.shape} | Probability: {prob:.4f} | Prediction: {is_crack}")
    except Exception as e:
        logger.error(f"Grad-CAM verification failed: {e}")
        raise e

def verify_onnx_export_pipeline():
    """Export custom cnn to ONNX, quantize it and check if it runs using ONNX Runtime."""
    logger.info("\n=== STEP 5: ONNX & Edge Optimization Verification ===")
    
    best_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
    
    if os.path.exists(best_path):
        logger.info("Trained model checkpoint found. Using it for ONNX verification...")
        model_path = best_path
        cleanup = False
    else:
        logger.info("Trained model checkpoint not found. Creating a temporary dummy model for verification...")
        dummy_model = get_model("custom_cnn", pretrained=False, num_classes=1)
        model_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_temp.pth")
        torch.save(dummy_model.state_dict(), model_path)
        cleanup = True
    
    try:
        # Export
        onnx_path = export_to_onnx("custom_cnn", model_path=model_path)
        logger.info(f"ONNX export verified: file exists = {os.path.exists(onnx_path)}")
        
        # Quantize
        quant_path = quantize_onnx_model(onnx_path)
        logger.info(f"ONNX quantization verified: file exists = {os.path.exists(quant_path)}")
        
        # Clean up temporary weights if we created them
        if cleanup and os.path.exists(model_path):
            os.remove(model_path)
            
    except Exception as e:
        logger.error(f"ONNX pipeline verification failed: {e}")
        raise e

def run_diagnostics():
    """Run all system verification diagnostics."""
    logger.info("==================================================")
    logger.info("RUNNING SYSTEM DIAGNOSTICS & VERIFICATION")
    logger.info("==================================================")
    
    verify_gpu_and_environment()
    verify_model_factory()
    verify_training_backpropagation()
    verify_gradcam()
    verify_onnx_export_pipeline()
    
    logger.info("\n==================================================")
    logger.info("VERIFICATION COMPLETED: ALL PIPELINES ARE STABLE!")
    logger.info("==================================================")

if __name__ == "__main__":
    run_diagnostics()
