import torch
from torchvision import transforms
from PIL import Image

def get_model_transform() -> transforms.Compose:
    """
    Returns the exact canonical preprocessing transform used by the NTIRE baseline model.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for EfficientNet-B0 inference.
    
    Args:
        image (PIL.Image.Image): The input image.
        
    Returns:
        torch.Tensor: Normalized image tensor ready for the model,
                      with a batch dimension added (1, C, H, W).
    """
    # Convert image to RGB to ensure 3 channels
    image = image.convert("RGB")
    
    preprocess = get_model_transform()
    
    # Apply preprocessing
    tensor = preprocess(image)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    batch_tensor = tensor.unsqueeze(0)
    
    return batch_tensor
