import torch
import torch.nn as nn
from models.custom_cnn import CustomFastCNN
from models.mobilenet_v2 import get_mobilenet_v2
from models.efficientnet import get_efficientnet_b0
from models.rcnn import get_rcnn

def get_model(
    model_name: str, 
    pretrained: bool = True, 
    num_classes: int = 1, 
    freeze_backbone: bool = False
) -> nn.Module:
    """Factory function to instantiate models by name."""
    model_name = model_name.lower().strip()
    
    if model_name == "custom_cnn":
        # Custom CNN does not use pretrained weights
        return CustomFastCNN(num_classes=num_classes)
        
    elif model_name == "mobilenet_v2":
        return get_mobilenet_v2(
            pretrained=pretrained, 
            num_classes=num_classes, 
            freeze_backbone=freeze_backbone
        )
        
    elif model_name == "efficientnet":
        return get_efficientnet_b0(
            pretrained=pretrained, 
            num_classes=num_classes, 
            freeze_backbone=freeze_backbone
        )
        
    elif model_name == "rcnn":
        return get_rcnn(
            pretrained=pretrained,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )
        
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            "Supported models: 'custom_cnn', 'mobilenet_v2', 'efficientnet', 'rcnn'"
        )

if __name__ == "__main__":
    # Quick test of model factory
    for name in ["custom_cnn", "mobilenet_v2", "efficientnet", "rcnn"]:
        m = get_model(name, pretrained=False)
        print(f"Factory successfully loaded: {name} (Params: {sum(p.numel() for p in m.parameters() if p.requires_grad)})")
