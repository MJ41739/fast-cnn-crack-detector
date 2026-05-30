import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from backend.config import settings
from models.factory import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def export_to_onnx(model_name: str, model_path: str = None) -> str:
    """Export the trained PyTorch model to ONNX format."""
    device = torch.device("cpu")  # ONNX export is safest on CPU
    
    # Load model architecture
    model = get_model(model_name, pretrained=False, num_classes=1)
    
    if model_path is None:
        model_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_best.pth")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first!")
        
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dummy input matching the model input shape: batch_size=1, channels=3, h=224, w=224
    dummy_input = torch.randn(1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE, requires_grad=False)
    
    onnx_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}.onnx")
    
    logger.info(f"Exporting PyTorch model '{model_name}' to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Enable dynamic batch size for flexible backend batch inference
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model successfully exported and verified at: {onnx_path}")
    return onnx_path

def quantize_onnx_model(onnx_path: str) -> str:
    """Quantize the exported ONNX model to INT8 dynamic quantization."""
    quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")
    logger.info(f"Quantizing ONNX model: {onnx_path} -> {quantized_path}")
    
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QUInt8
    )
    
    logger.info(f"Quantized ONNX model saved to: {quantized_path}")
    # Compare file sizes
    orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
    logger.info(f"Original size: {orig_size:.2f} MB | Quantized size: {quant_size:.2f} MB (Compression: {orig_size/quant_size:.1f}x)")
    return quantized_path

def apply_pruning(model_name: str, pruning_amount: float = 0.3, model_path: str = None) -> str:
    """Apply L1 Unstructured Pruning to all Conv2D layers in the model and save the weights."""
    device = torch.device("cpu")
    model = get_model(model_name, pretrained=False, num_classes=1)
    
    if model_path is None:
        model_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_best.pth")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    logger.info(f"Applying L1 Unstructured Pruning ({pruning_amount:.1%}) to Conv2D layers...")
    
    # Iterate through modules and apply pruning to Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
            # Make pruning permanent by removing reparameterization hooks
            prune.remove(module, "weight")
            
    pruned_path = os.path.join(settings.CHECKPOINT_DIR, f"{model_name}_pruned.pth")
    torch.save(model.state_dict(), pruned_path)
    logger.info(f"Pruned model weights saved to: {pruned_path}")
    return pruned_path

def convert_to_tflite_instructions(model_name: str):
    """Print guidelines/instructions and auto-setup convert flow for PyTorch -> ONNX -> TF -> TFLite."""
    logger.info("="*80)
    logger.info("TFLITE EDGE CONVERSION DEPLOYMENT WORKFLOW")
    logger.info("="*80)
    logger.info("To deploy the model on lightweight/mobile edge devices using TFLite, use the following pipeline:")
    logger.info("1. Export PyTorch to ONNX (Run: python training/export.py --model custom_cnn)")
    logger.info("2. Convert ONNX to TensorFlow Graph using 'onnx-tf' or 'onnx2tf':")
    logger.info("   pip install onnx-tf tensorflow")
    logger.info("   onnx-tf convert -i checkpoints/custom_cnn.onnx -o checkpoints/tf_model")
    logger.info("3. Convert TensorFlow SavedModel to TFLite:")
    logger.info("   import tensorflow as tf")
    logger.info("   converter = tf.lite.TFLiteConverter.from_saved_model('checkpoints/tf_model')")
    logger.info("   tflite_model = converter.convert()")
    logger.info("   with open('checkpoints/custom_cnn.tflite', 'wb') as f:")
    logger.info("       f.write(tflite_model)")
    logger.info("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and Optimize Models for Edge Deployment")
    parser.add_argument(
        "--model", 
        type=str, 
        default="custom_cnn", 
        choices=["custom_cnn", "mobilenet_v2", "efficientnet"],
        help="Model to optimize and export"
    )
    parser.add_argument("--prune", type=float, default=0.3, help="Amount of weights to prune (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    try:
        # 1. Export PyTorch to ONNX
        onnx_path = export_to_onnx(args.model)
        
        # 2. Quantize ONNX model to INT8
        quant_path = quantize_onnx_model(onnx_path)
        
        # 3. Apply pruning to PyTorch model weights
        prune_path = apply_pruning(args.model, args.prune)
        
        # 4. Show instructions for TFLite conversion
        convert_to_tflite_instructions(args.model)
        
    except Exception as e:
        logger.error(f"Error during export/optimization: {e}")
