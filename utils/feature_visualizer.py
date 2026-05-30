import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings
from models.factory import get_model
from utils.gradcam import get_gradcam_visualization
from training.dataset import get_dataloaders

def generate_visualizations(image_path: str = None):
    print("=== FEATURE VISUALIZATION ===")
    device = torch.device(settings.DEVICE)
    best_path = os.path.join(settings.CHECKPOINT_DIR, "custom_cnn_best.pth")
    
    if not os.path.exists(best_path):
        print(f"Error: Trained model weights not found at {best_path}!")
        return
        
    # Load model
    model = get_model("custom_cnn", pretrained=False, num_classes=1).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    
    # Load a sample cracked image
    if image_path is None:
        train_loader, val_loader, test_loader = get_dataloaders()
        test_dataset = test_loader.dataset
        # Find first crack image
        pos_idx = next(i for i, label in enumerate(test_dataset.labels) if label == 1)
        image_path = test_dataset.file_paths[pos_idx]
        
    print(f"Using image: {image_path}")
    orig_image = Image.open(image_path).convert("RGB")
    
    # Define standard transform (without noise or augmentation)
    preprocess = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(orig_image).unsqueeze(0).to(device)
    
    # 1. Grad-CAM Visualization
    print("Generating Grad-CAM visualization...")
    overlay, prob, is_crack = get_gradcam_visualization(model, orig_image, device)
    
    gradcam_save_path = os.path.join(settings.OUTPUT_DIR, "feature_vis_gradcam.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM Heatmap\nProb: {prob:.4f} | Prediction: {'Crack' if is_crack else 'No Crack'}", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.savefig(gradcam_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM to {gradcam_save_path}")
    
    # 2. Intermediate Activation Maps / Layer Outputs
    print("Extracting intermediate layer outputs...")
    
    # Dictionary to store intermediate outputs
    activations = {}
    
    # Define hooks to capture outputs of intermediate layers
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
        
    # Register hooks on specific layers
    # Initial Conv
    hook0 = model.features[0].register_forward_hook(get_activation("conv1"))
    # Depthwise separable blocks
    hook1 = model.features[3].register_forward_hook(get_activation("block1"))
    hook2 = model.features[6].register_forward_hook(get_activation("block2"))
    hook3 = model.features[9].register_forward_hook(get_activation("block3"))
    hook4 = model.features[12].register_forward_hook(get_activation("block4"))
    
    # Run forward pass
    with torch.no_grad():
        _ = model(input_tensor)
        
    # Remove hooks
    hook0.remove()
    hook1.remove()
    hook2.remove()
    hook3.remove()
    hook4.remove()
    
    # Plot feature maps
    for name, act in activations.items():
        # act has shape [1, C, H, W]
        act_maps = act[0].numpy()
        num_channels = act_maps.shape[0]
        
        # Select up to 16 channels to display in a 4x4 grid
        n_features = min(16, num_channels)
        grid_size = int(np.ceil(np.sqrt(n_features)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        fig.suptitle(f"Activation Maps for Layer: {name} (Shape: {list(act.shape)})", fontsize=14, fontweight="bold")
        
        for idx in range(grid_size * grid_size):
            ax = axes[idx // grid_size, idx % grid_size]
            if idx < n_features:
                # Get map and normalize it for visualization
                f_map = act_maps[idx]
                f_map_min, f_map_max = f_map.min(), f_map.max()
                if f_map_max > f_map_min:
                    f_map = (f_map - f_map_min) / (f_map_max - f_map_min)
                ax.imshow(f_map, cmap="viridis")
            ax.axis("off")
            
        plt.tight_layout()
        save_path = os.path.join(settings.OUTPUT_DIR, f"feature_vis_{name}_maps.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {name} activation maps to {save_path}")
        
    # 3. Overall Feature Summary
    # Combine original image, Grad-CAM, and some channel averages
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Feature Visualization Summary & Crack Feature Localization", fontsize=16, fontweight="bold")
    
    # Original Image
    img_resized = orig_image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title("Original Image (224x224)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    
    # Grad-CAM Overlay
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Grad-CAM Heatmap (Prob: {prob:.4f})", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    
    # Conv1 Average
    conv1_avg = activations["conv1"][0].mean(dim=0).numpy()
    axes[0, 2].imshow(conv1_avg, cmap="magma")
    axes[0, 2].set_title("Conv1 Channel Average (Low-level Edges)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    
    # Block 1 Average
    block1_avg = activations["block1"][0].mean(dim=0).numpy()
    axes[1, 0].imshow(block1_avg, cmap="magma")
    axes[1, 0].set_title("Block 1 Average (Mid-level Features)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    
    # Block 2 Average
    block2_avg = activations["block2"][0].mean(dim=0).numpy()
    axes[1, 1].imshow(block2_avg, cmap="magma")
    axes[1, 1].set_title("Block 2 Average (Deep Textures)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    
    # Block 4 Average
    block4_avg = activations["block4"][0].mean(dim=0).numpy()
    axes[1, 2].imshow(block4_avg, cmap="magma")
    axes[1, 2].set_title("Block 4 Average (High-level Semantic)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    summary_save_path = os.path.join(settings.OUTPUT_DIR, "feature_vis_summary.png")
    plt.savefig(summary_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature visualization summary to {summary_save_path}")

if __name__ == "__main__":
    generate_visualizations()
