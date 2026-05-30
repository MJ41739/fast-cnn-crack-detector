import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Add project root to python path
sys.path.append(r"c:\Users\Mayur Jadhav\OneDrive\Desktop\FASTCNN")
from backend.config import settings
from models.factory import get_model
from training.dataset import get_dataloaders
from training.export import export_to_onnx, quantize_onnx_model

def fix_checkpoints():
    print("=== STEP 1: RESTORING TRAINED CHECKPOINTS ===")
    checkpoint_dir = settings.CHECKPOINT_DIR
    latest_path = os.path.join(checkpoint_dir, "custom_cnn_latest.pth")
    best_path = os.path.join(checkpoint_dir, "custom_cnn_best.pth")
    
    if not os.path.exists(latest_path):
        print(f"ERROR: Latest checkpoint not found at {latest_path}!")
        return False
        
    print(f"Loading trained weights from {latest_path}...")
    checkpoint = torch.load(latest_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]
    
    # Save as custom_cnn_best.pth
    torch.save(model_state, best_path)
    print(f"Saved trained weights to {best_path}.")
    
    # Re-export to ONNX and quantize using the actual trained weights
    print("\nRe-exporting trained weights to ONNX...")
    try:
        onnx_path = export_to_onnx("custom_cnn", model_path=best_path)
        print(f"Successfully re-exported ONNX to {onnx_path}.")
        
        quant_path = quantize_onnx_model(onnx_path)
        print(f"Successfully re-exported quantized ONNX to {quant_path}.")
        return True
    except Exception as e:
        print(f"ONNX export/quantization failed: {e}")
        return False

def test_inference_pipeline():
    print("\n=== STEP 2: TESTING INFERENCE PIPELINE ===")
    device = torch.device(settings.DEVICE)
    best_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
    
    # Load model
    model = get_model("custom_cnn", pretrained=False, num_classes=1).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    # Get 10 positive and 10 negative test images
    
    # Get 10 positive and 10 negative test images
    train_loader, val_loader, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset
    
    pos_indices = [i for i, label in enumerate(test_dataset.labels) if label == 1][:10]
    neg_indices = [i for i, label in enumerate(test_dataset.labels) if label == 0][:10]
    
    print("\nEvaluating 20 test images:")
    print(f"{'Filename':<30} | {'Ground Truth':<12} | {'Logit':<8} | {'Probability':<12} | {'Prediction':<15}")
    print("-" * 88)
    
    for idx in pos_indices + neg_indices:
        img_path = test_dataset.file_paths[idx]
        gt_label = "Crack" if test_dataset.labels[idx] == 1 else "No Crack"
        
        img = Image.open(img_path).convert("RGB")
        tensor = test_dataset.transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = model(tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            pred_class = "Crack" if prob >= 0.5 else "No Crack"
            
        print(f"{os.path.basename(img_path):<30} | {gt_label:<12} | {logit:+.4f} | {prob:.4f} ({prob*100:.1f}%) | {pred_class:<15}")

def run_diagnostic_tests():
    print("\n=== STEP 3: RUNNING DIAGNOSTIC TESTS ===")
    device = torch.device(settings.DEVICE)
    best_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
    
    model = get_model("custom_cnn", pretrained=False, num_classes=1).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    train_loader, val_loader, test_loader = get_dataloaders()
    test_dataset = test_loader.dataset
    
    # Helper to print outputs
    def eval_tensors(tensors, title):
        print(f"\n{title}:")
        with torch.no_grad():
            logits = model(tensors).squeeze(-1)
            probs = torch.sigmoid(logits)
            for i in range(len(tensors)):
                prob = probs[i].item()
                pred = "Crack" if prob >= 0.5 else "No Crack"
                print(f"  Sample {i+1}: Logit = {logits[i].item():+.4f}, Prob = {prob:.4f} ({prob*100:.1f}%), Pred = {pred}")

    # Find one positive and one negative image
    pos_idx = next(i for i, label in enumerate(test_dataset.labels) if label == 1)
    neg_idx = next(i for i, label in enumerate(test_dataset.labels) if label == 0)
    
    pos_path = test_dataset.file_paths[pos_idx]
    neg_path = test_dataset.file_paths[neg_idx]
    
    # Test A: One positive image repeated 10 times
    pos_img = Image.open(pos_path).convert("RGB")
    pos_tensor = test_dataset.transform(pos_img).unsqueeze(0).to(device)
    pos_batch = pos_tensor.repeat(10, 1, 1, 1)
    eval_tensors(pos_batch, f"Test A (1 Positive image repeated 10 times - {os.path.basename(pos_path)})")
    
    # Test B: One negative image repeated 10 times
    neg_img = Image.open(neg_path).convert("RGB")
    neg_tensor = test_dataset.transform(neg_img).unsqueeze(0).to(device)
    neg_batch = neg_tensor.repeat(10, 1, 1, 1)
    eval_tensors(neg_batch, f"Test B (1 Negative image repeated 10 times - {os.path.basename(neg_path)})")
    
    # Test C: Random noise image
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    noise_img = torch.rand(3, 224, 224).to(device) # values [0, 1]
    noise_normalized = (noise_img - imagenet_mean) / imagenet_std
    noise_batch = noise_normalized.unsqueeze(0).repeat(5, 1, 1, 1)
    eval_tensors(noise_batch, "Test C (Random noise image repeated 5 times)")
    
    # Test D: Completely black image
    black_img = torch.zeros(3, 224, 224).to(device)
    black_normalized = (black_img - imagenet_mean) / imagenet_std
    black_batch = black_normalized.unsqueeze(0).repeat(5, 1, 1, 1)
    eval_tensors(black_batch, "Test D (Completely black image repeated 5 times)")
    
    # Test E: Completely white image
    white_img = torch.ones(3, 224, 224).to(device)
    white_normalized = (white_img - imagenet_mean) / imagenet_std
    white_batch = white_normalized.unsqueeze(0).repeat(5, 1, 1, 1)
    eval_tensors(white_batch, "Test E (Completely white image repeated 5 times)")

if __name__ == "__main__":
    if fix_checkpoints():
        test_inference_pipeline()
        run_diagnostic_tests()
