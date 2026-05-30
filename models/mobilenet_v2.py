import torch
import torch.nn as nn
import torchvision.models as models

def get_mobilenet_v2(pretrained: bool = True, num_classes: int = 1, freeze_backbone: bool = False) -> nn.Module:
    """Instantiate MobileNetV2 with customized classifier head."""
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2()

    # Freeze feature layers if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Get input size of the classifier layer
    in_features = model.classifier[1].in_features

    # Replace classifier head
    # MobileNetV2 classifier structure is Sequential(Dropout, Linear)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)  # Outputs raw logit
    )
    
    return model

if __name__ == "__main__":
    # Test model with dummy input
    model = get_mobilenet_v2(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("MobileNetV2 output shape:", out.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
