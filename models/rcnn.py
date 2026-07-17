import torch
import torch.nn as nn

class RCL(nn.Module):
    """Recurrent Convolutional Layer (RCL) block based on Liang & Hu (2015).
    
    Integrates feed-forward and intra-layer recurrent connections to refine spatial features.
    """
    def __init__(self, in_channels: int, out_channels: int, steps: int = 3):
        super().__init__()
        self.steps = steps
        # Feed-forward convolution: projects input feature map to out_channels
        self.conv_feedforward = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # Recurrent convolution: shares weights across time steps
        self.conv_recurrent = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feedforward calculation
        ff = self.conv_feedforward(x)
        # Initial step
        h = self.relu(self.bn(ff))
        # Recurrent cycles
        for _ in range(self.steps - 1):
            h = self.relu(self.bn(ff + self.conv_recurrent(h)))
        return h

class RecurrentCNN(nn.Module):
    """Recurrent Convolutional Neural Network (RCNN) for binary crack classification."""
    def __init__(self, num_classes: int = 1, steps: int = 3):
        super().__init__()
        
        # Initial standard convolution to extract base spatial features
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # RCL Block 1
            RCL(32, 64, steps=steps),                                         # 64 x 112 x 112
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 64 x 56 x 56
            nn.Dropout2d(0.2),
            
            # RCL Block 2
            RCL(64, 128, steps=steps),                                        # 128 x 56 x 56
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 128 x 28 x 28
            nn.Dropout2d(0.2),
            
            # RCL Block 3
            RCL(128, 256, steps=steps),                                       # 256 x 28 x 28
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 256 x 14 x 14
            nn.Dropout2d(0.3),
            
            # RCL Block 4
            RCL(256, 512, steps=steps),                                       # 512 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 512 x 7 x 7
            nn.Dropout2d(0.3),
        )
        
        # Global Average Pooling (reduces 512x7x7 to 512x1x1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)  # Outputs raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_rcnn(pretrained: bool = False, num_classes: int = 1, freeze_backbone: bool = False) -> nn.Module:
    """Instantiate Recurrent CNN model.
    
    Note: RCNN is initialized from scratch as pretrained weights are not standardly available.
    """
    model = RecurrentCNN(num_classes=num_classes)
    
    if freeze_backbone:
        for name, param in model.features.named_parameters():
            param.requires_grad = False
            
    return model

if __name__ == "__main__":
    # Test model with dummy input
    model = get_rcnn()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("RecurrentCNN output shape:", out.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
