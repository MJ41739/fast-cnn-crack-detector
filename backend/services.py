import os
import io
import time
import base64
import logging
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import onnxruntime as ort

from backend.config import settings
from models.factory import get_model
from utils.gradcam import get_gradcam_visualization

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class InferenceService:
    """Core class that manages model loading, caching, and running predictions."""
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.active_model_name = settings.ACTIVE_MODEL_NAME
        
        # Internal storage for models
        self.pytorch_model = None
        self.onnx_session = None
        
        # Standard preprocessing transform (ImageNet normalization)
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])
        
        # Load default model
        self.load_model(self.active_model_name)

    def load_model(self, model_name: str):
        """Load a PyTorch checkpoint or an ONNX session based on model name."""
        model_name_clean = model_name.lower().strip()
        logger.info(f"Loading active inference model: '{model_name_clean}'...")
        
        # Clear existing models from memory
        self.pytorch_model = None
        self.onnx_session = None
        
        if "onnx" in model_name_clean:
            # ONNX Inference Setup
            if model_name_clean == "onnx_quantized":
                onnx_file = "custom_cnn_quantized.onnx"
            else:
                onnx_file = "custom_cnn.onnx"
                
            onnx_path = os.path.join(settings.CHECKPOINT_DIR, onnx_file)
            
            # Fallback if optimized ONNX is not built yet
            if not os.path.exists(onnx_path):
                # Try standard onnx file first
                fallback_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn.onnx")
                if os.path.exists(fallback_path):
                    logger.warning(f"Quantized ONNX not found. Falling back to standard ONNX: {fallback_path}")
                    onnx_path = fallback_path
                else:
                    raise FileNotFoundError(
                        f"ONNX model file not found at: {onnx_path}. "
                        "Please run 'python training/export.py --model custom_cnn' to generate it first."
                    )
            
            # Setup ONNX Runtime session
            # Use CUDA provider if GPU available, fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if settings.DEVICE == "cuda" else ['CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"ONNX session loaded successfully from {onnx_path} on providers {self.onnx_session.get_providers()}")
            
        else:
            # PyTorch Inference Setup
            # Create model instance
            model = get_model(model_name_clean, pretrained=False, num_classes=1)
            checkpoint_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name_clean}_best.pth")
            
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                logger.info(f"Loaded trained PyTorch weights from: {checkpoint_path}")
            else:
                logger.warning(
                    f"Trained checkpoint not found at {checkpoint_path}. "
                    "Inference will run on raw (untrained) model architecture!"
                )
                
            model = model.to(self.device)
            model.eval()
            self.pytorch_model = model
            
        self.active_model_name = model_name_clean
        logger.info(f"Active model successfully set to: {self.active_model_name}")
        self.pre_warm()

    def pre_warm(self):
        """Run a dummy forward pass to warm up PyTorch/ONNX engines."""
        try:
            logger.info("Pre-warming active model for inference...")
            # Create a dummy preprocessed tensor: [1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE]
            dummy_tensor = torch.zeros((1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE))
            
            if self.onnx_session is not None:
                input_name = self.onnx_session.get_inputs()[0].name
                self.onnx_session.run(None, {input_name: dummy_tensor.numpy()})
            else:
                dummy_tensor = dummy_tensor.to(self.device)
                with torch.no_grad():
                    self.pytorch_model(dummy_tensor)
            logger.info("Model pre-warming completed successfully!")
        except Exception as e:
            logger.warning(f"Failed to pre-warm model: {e}")

    def preprocess_image(self, image_bytes: bytes) -> tuple[Image.Image, torch.Tensor]:
        """Convert raw bytes to PIL Image and preprocessed tensor."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
        return image, tensor

    def predict(self, image_bytes: bytes, filename: str, generate_heatmap: bool = True) -> dict:
        """Run inference on the active model, compute confidence, and generate base64 Grad-CAM."""
        start_time = time.perf_counter()
        
        # Preprocess
        pil_image, input_tensor = self.preprocess_image(image_bytes)
        
        # Running inference
        if self.onnx_session is not None:
            # ONNX Inference
            input_name = self.onnx_session.get_inputs()[0].name
            # Format inputs as dict
            onnx_inputs = {input_name: input_tensor.numpy()}
            outputs = self.onnx_session.run(None, onnx_inputs)
            logit = float(outputs[0][0][0])
            
            # Sigmoid activation
            prob = 1.0 / (1.0 + np.exp(-logit))
            is_crack = prob >= 0.5
            confidence = prob if is_crack else (1.0 - prob)
            
            inference_time = (time.perf_counter() - start_time) * 1000.0  # in ms
            
            # Grad-CAM is not natively supported for compiled ONNX models in this script,
            # so we either run Grad-CAM using the PyTorch CustomFastCNN equivalent or skip it.
            # We will run it on the PyTorch CustomFastCNN (loading weights if available) to display heatmaps even for ONNX models!
            heatmap_base64 = None
            if generate_heatmap:
                try:
                    # Instantiating a temporary CustomFastCNN to extract Grad-CAM maps
                    temp_model = get_model("custom_cnn", pretrained=False)
                    temp_weights = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
                    if os.path.exists(temp_weights):
                        temp_model.load_state_dict(torch.load(temp_weights, map_location=self.device))
                    temp_model = temp_model.to(self.device)
                    
                    overlay, _, _ = get_gradcam_visualization(temp_model, pil_image, self.device)
                    heatmap_base64 = self._image_to_base64(overlay)
                except Exception as e:
                    logger.warning(f"Grad-CAM overlay failed during ONNX session inference: {e}")
                    
        else:
            # PyTorch Inference
            input_tensor = input_tensor.to(self.device)
            
            if generate_heatmap:
                # Runs forward pass, backward pass, extracts activations and blends heatmap
                overlay, prob, is_crack = get_gradcam_visualization(self.pytorch_model, pil_image, self.device)
                heatmap_base64 = self._image_to_base64(overlay)
            else:
                with torch.no_grad():
                    output = self.pytorch_model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    is_crack = prob >= 0.5
                heatmap_base64 = None
                
            confidence = prob if is_crack else (1.0 - prob)
            inference_time = (time.perf_counter() - start_time) * 1000.0  # in ms
            
        prediction_str = "Crack Detected" if is_crack else "No Crack Detected"
        
        return {
            "filename": filename,
            "prediction": prediction_str,
            "confidence": confidence,
            "probability": prob,
            "heatmap_image": heatmap_base64,
            "inference_time_ms": inference_time
        }

    def predict_batch(self, images_list: list[tuple[bytes, str]]) -> dict:
        """Run batched predictions for high efficiency, optimizing processing speed."""
        start_time = time.perf_counter()
        predictions = []
        
        # For batching, we group them into a single tensor
        tensors = []
        valid_images = []
        
        for bytes_data, filename in images_list:
            try:
                pil_img, tensor = self.preprocess_image(bytes_data)
                tensors.append(tensor)
                valid_images.append((pil_img, filename))
            except Exception as e:
                logger.error(f"Failed to preprocess {filename} during batch run: {e}")
                predictions.append({
                    "filename": filename,
                    "prediction": "Processing Error",
                    "confidence": 0.0,
                    "probability": 0.0,
                    "heatmap_image": None,
                    "inference_time_ms": 0.0
                })
                
        if not tensors:
            return {
                "total_images": len(images_list),
                "predictions": predictions,
                "average_latency_ms": 0.0
            }
            
        # Stack all tensors into a single batch tensor: [B, 3, 224, 224]
        batch_tensor = torch.cat(tensors, dim=0)
        batch_size = batch_tensor.shape[0]
        
        probs = []
        
        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: batch_tensor.numpy()})
            logits = outputs[0].flatten()
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            batch_tensor = batch_tensor.to(self.device)
            self.pytorch_model.eval()
            with torch.no_grad():
                logits = self.pytorch_model(batch_tensor).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
        # Fill response details
        # For batching, we skip Grad-CAM heatmap generation to ensure low inference latency (<100ms targets)
        batch_latency = (time.perf_counter() - start_time) * 1000.0
        avg_latency = batch_latency / batch_size
        
        for idx, (pil_img, filename) in enumerate(valid_images):
            prob = float(probs[idx])
            is_crack = prob >= 0.5
            confidence = prob if is_crack else (1.0 - prob)
            prediction_str = "Crack Detected" if is_crack else "No Crack Detected"
            
            predictions.append({
                "filename": filename,
                "prediction": prediction_str,
                "confidence": confidence,
                "probability": prob,
                "heatmap_image": None,
                "inference_time_ms": avg_latency
            })
            
        return {
            "total_images": len(images_list),
            "predictions": predictions,
            "average_latency_ms": avg_latency
        }

    def _image_to_base64(self, img_array: np.ndarray) -> str:
        """Convert a numpy RGB image array to base64 Data URL string."""
        pil_img = Image.fromarray(img_array.astype('uint8'))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

# Create a singleton service instance
inference_service = InferenceService()
