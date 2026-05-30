import torch
import torch.nn as nn
import torchvision.models as models

def get_efficientnet_b0(pretrained: bool = True, num_classes: int = 1, freeze_backbone: bool = False) -> nn.Module:
    """Instantiate EfficientNetB0 with customized classifier head."""
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0()

    # Freeze feature layers if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Get input size of the classifier layer
    in_features = model.classifier[1].in_features

    # Replace classifier head
    # EfficientNet classifier structure is Sequential(Dropout, Linear)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)  # Outputs raw logit
    )
    
    return model

if __name__ == "__main__":
    # Test model with dummy input
    model = get_efficientnet_b0(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("EfficientNetB0 output shape:", out.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
