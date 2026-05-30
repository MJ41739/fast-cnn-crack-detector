import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict

from backend.config import settings

def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Recursively search for the last Conv2d layer in a model."""
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in the model.")

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation."""
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        
        # If no target layer provided, auto-detect the final Conv2d layer
        self.target_layer = target_layer if target_layer else find_last_conv_layer(model)
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """Generate raw heatmap weights for a given input tensor."""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Since it is a binary classifier, we take output score directly
        if output.shape[1] == 1:
            score = output[0]
        else:
            score = output[0, class_idx]
            
        # Backward pass
        score.backward()
        
        # Calculate gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global Average Pooling of gradients (weights alpha)
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=0)  # [H, W]
        
        # Apply ReLU to retain only positive features (features that positively impact classification)
        cam = torch.clamp(cam, min=0)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam - cam_min
            
        return cam.cpu().numpy()

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

def apply_gradcam_overlay(
    image_path_or_pil: Image.Image, 
    heatmap: np.ndarray, 
    alpha: float = 0.5
) -> np.ndarray:
    """Blend the Grad-CAM heatmap with the original image using OpenCV."""
    # Convert PIL Image or path to OpenCV BGR
    if isinstance(image_path_or_pil, Image.Image):
        orig_img = np.array(image_path_or_pil)
        # Convert RGB to BGR for cv2
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    else:
        orig_img = cv2.imread(image_path_or_pil)
        if orig_img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path_or_pil}")
            
    h, w, _ = orig_img.shape
    
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Scale heatmap to [0, 255] and convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET color map (turns high values red, low values blue)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Overlay heatmap onto the original image
    overlay = cv2.addWeighted(orig_img, 1.0 - alpha, heatmap_color, alpha, 0)
    
    # Convert back to RGB for PIL/matplotlib
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb

def get_gradcam_visualization(
    model: nn.Module, 
    image: Image.Image, 
    device: torch.device
) -> Tuple[np.ndarray, float, bool]:
    """Helper function to run Grad-CAM on a single PIL image and return visualization."""
    # Base transforms (excluding data augmentations like flip/noise)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    preprocess = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Preprocess image
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Run Grad-CAM
    cam_generator = GradCAM(model)
    try:
        # Run forward and backward pass
        heatmap = cam_generator.generate_heatmap(input_tensor)
        
        # Get raw probability prediction
        model.eval()
        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
            is_crack = prob >= 0.5
            
        # Apply overlay on the original PIL image (resized to standard settings if needed)
        image_resized = image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        overlay = apply_gradcam_overlay(image_resized, heatmap)
        
        return overlay, prob, is_crack
    finally:
        cam_generator.remove_hooks()
