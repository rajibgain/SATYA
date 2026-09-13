import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import urllib.error

def create_model(pretrained=True, freeze_backbone=False):
    """
    Creates a ConvNeXt-Tiny model for binary classification (0=real, 1=ai_generated).
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    
    try:
        model = convnext_tiny(weights=weights)
    except Exception as e:
        # Fallback if download fails (e.g. offline during smoke test)
        print(f"Warning: Failed to load pretrained weights ({e}). Initializing randomly.")
        model = convnext_tiny(weights=None)
        
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the classifier for binary classification
    # ConvNeXt classifier is a sequential block. We replace the final linear layer.
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)
    
    # Ensure classifier requires grad even if backbone is frozen
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model

def unfreeze_stages(model, n_stages):
    """
    ConvNeXt-Tiny has 4 stages in `model.features`.
    n_stages = 0: Only classifier is trainable.
    n_stages = 1: Unfreeze stage 4 (and its downsample layer).
    n_stages = 2: Unfreeze stages 3 and 4.
    n_stages = 3: Unfreeze stages 2, 3, and 4.
    n_stages = 4: Unfreeze everything.
    """
    stages_to_unfreeze = []
    if n_stages >= 1:
        stages_to_unfreeze.extend([6, 7]) # downsample 4, stage 4
    if n_stages >= 2:
        stages_to_unfreeze.extend([4, 5]) # downsample 3, stage 3
    if n_stages >= 3:
        stages_to_unfreeze.extend([2, 3]) # downsample 2, stage 2
    if n_stages >= 4:
        stages_to_unfreeze.extend([0, 1]) # stem, stage 1
        
    for idx in stages_to_unfreeze:
        for param in model.features[idx].parameters():
            param.requires_grad = True
            
    # Always ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model
