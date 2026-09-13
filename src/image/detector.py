import os
from pathlib import Path
import torch
import torchvision.models as models

# Configuration: default path to the trained model weights
DEFAULT_MODEL_PATH = Path("models") / "image" / "best_image_model.pth"

class ImageForensicsDetector:
    """
    Image Forensics Detector for identifying deepfakes and AI-generated images.
    Uses EfficientNet-B0 architecture with a custom trained head for binary classification.
    """
    
    # MUST match ImageFolder's training mapping exactly
    CLASS_NAMES = {0: "Real", 1: "Fake"}
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        Initialize the detector.
        
        Args:
            model_path (str or Path): Path to the trained SATYA image model weights.
        """
        self.model_path = Path(model_path)
        # Automatically use CUDA when available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """
        Load the model architecture and custom weights.
        Raises a controlled error if the custom model weights are not found.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"The trained SATYA image model is not available yet at '{self.model_path}'.\n"
                "Please train the model first or provide a valid path to the trained weights.\n"
                "Note: We do NOT fall back to ImageNet classification for deepfake detection."
            )
            
        try:
            # Initialize EfficientNet-B0 architecture without pretrained ImageNet weights
            self.model = models.efficientnet_b0(weights=None)
            
            # Modify the classifier to match our 2-class problem (e.g., Fake vs Real)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
            
            # Load the custom trained weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            
            # Move model to the selected device and put into evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load the model from {self.model_path}: {e}")

    def predict(self, input_data) -> dict:
        """
        Run inference on the image to detect if it is Fake or Real.
        
        Args:
            input_data (torch.Tensor or PIL.Image.Image): The input image or preprocessed tensor (1, C, H, W).
            
        Returns:
            dict: Structured prediction result containing label, confidence, and device.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot run prediction.")
            
        if not isinstance(input_data, torch.Tensor):
            # If a PIL Image is passed, use the shared canonical transform
            from src.image.preprocess import preprocess_image
            input_tensor = preprocess_image(input_data)
        else:
            input_tensor = input_data
            
        # Ensure the input is on the correct device
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Convert raw logits to probabilities
            probabilities = outputs.squeeze(0).softmax(0)
            
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        
        label = self.CLASS_NAMES.get(predicted_class_idx.item(), "Unknown")
        
        return {
            "label": label,
            "class_idx": predicted_class_idx.item(),
            "confidence": float(confidence.item()),
            "device": str(self.device.type)
        }
