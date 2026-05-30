import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block to optimize speed and parameters."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Depthwise layer: group convolutions per channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        # Pointwise layer: 1x1 convolution to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CustomFastCNN(nn.Module):
    """A highly optimized, lightweight Custom CNN for Edge/Fast Crack Detection."""
    def __init__(self, num_classes: int = 1):
        super().__init__()
        # Input: 3 x 224 x 224
        
        # Initial standard convolution to extract base spatial features
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Convolutional Block 1
            DepthwiseSeparableConv(32, 64),                                   # 64 x 112 x 112
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 64 x 56 x 56
            nn.Dropout2d(0.2),
            
            # Convolutional Block 2
            DepthwiseSeparableConv(64, 128),                                  # 128 x 56 x 56
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 128 x 28 x 28
            nn.Dropout2d(0.2),
            
            # Convolutional Block 3
            DepthwiseSeparableConv(128, 256),                                 # 256 x 28 x 28
            nn.MaxPool2d(kernel_size=2, stride=2),                           # 256 x 14 x 14
            nn.Dropout2d(0.3),
            
            # Convolutional Block 4
            DepthwiseSeparableConv(256, 512),                                 # 512 x 14 x 14
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

if __name__ == "__main__":
    # Test model with dummy input
    model = CustomFastCNN()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("CustomFastCNN output shape:", out.shape)
    print("Total Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
