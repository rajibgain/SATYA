import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class AIDetector(nn.Module):
    def __init__(self, freeze_blocks=True):
        super(AIDetector, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)
        
        # Replace the classifier for binary classification (Real vs AI)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 2)
        )
        
        if freeze_blocks:
            self.freeze_lower_layers()

    def freeze_lower_layers(self):
        """
        Freezes the stem and early stages of the network to preserve 
        low-level feature extractors, speeding up training and reducing overfitting.
        """
        # EfficientNet-B0 features has 8 main blocks. We freeze the first 4 blocks.
        for idx, child in enumerate(self.model.features.children()):
            if idx < 5:  # Stem + first few MBConv blocks
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test model initialization
    model = AIDetector(freeze_blocks=True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
