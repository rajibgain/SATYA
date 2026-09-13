import argparse
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models

# Import canonical preprocessing
from src.image.preprocess import preprocess_image

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Loads the EfficientNet-B0 model and restores the provided checkpoint.
    Recreates the exact architecture used during training.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    # Recreate the exact EfficientNet-B0 architecture
    model = models.efficientnet_b0(weights=None)
    
    # Recreate the exact classifier head for 2 classes (Real vs Fake)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Load the trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    return model

def analyze_image(image_path: str, model: nn.Module, device: torch.device, threshold: float = 0.50):
    """
    Runs inference on a single image and returns structured results.
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError:
        raise ValueError(f"Could not open or identify image at: {image_path}")
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")
        
    # Preprocess image using the exact canonical pipeline
    # preprocess_image already adds the batch dimension
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        
    # Class mapping from training: Real = 0, Fake = 1
    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    
    verdict = "LIKELY FAKE" if fake_prob >= threshold else "LIKELY REAL"
    
    return {
        "verdict": verdict,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "threshold": threshold,
        "model_name": "EfficientNet-B0"
    }

def main():
    parser = argparse.ArgumentParser(description="SATYA Image Inference")
    parser.add_argument("image_path", type=str, help="Path to the image to analyze")
    parser.add_argument("--model-path", type=str, default="models/image_ntire_unfrozen/best_image_model.pth", 
                        help="Path to the trained model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.50, 
                        help="Decision threshold for Fake class (default: 0.50)")
    
    args = parser.parse_args()
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(args.model_path, device)
        
        result = analyze_image(args.image_path, model, device, args.threshold)
        
        print("\n" + "=" * 40)
        print("SATYA IMAGE ANALYSIS")
        print("=" * 40)
        print(f"Verdict: {result['verdict']}")
        print(f"Fake Probability: {result['fake_probability'] * 100:.2f}%")
        print(f"Real Probability: {result['real_probability'] * 100:.2f}%")
        print(f"Decision Threshold: {result['threshold'] * 100:.0f}%")
        print(f"Model: {result['model_name']}")
        print("=" * 40 + "\n")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
