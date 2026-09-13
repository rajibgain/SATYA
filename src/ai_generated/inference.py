import argparse
import torch
from PIL import Image
from pathlib import Path

from src.ai_generated.model import create_model
from src.ai_generated.preprocess import get_eval_transforms

def run_inference(image_path, model_path, threshold=0.5, image_size=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(pretrained=False) # Will load weights from checkpoint
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Process image
    transform = get_eval_transforms(image_size)
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
        
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        prob_ai = torch.sigmoid(logits).item()
        
    prob_real = 1.0 - prob_ai
    
    verdict = "LIKELY AI-GENERATED" if prob_ai >= threshold else "LIKELY REAL"
    
    return {
        "model_name": "ConvNeXt-Tiny (AI-Generated)",
        "ai_generated_probability": prob_ai,
        "real_probability": prob_real,
        "verdict": verdict,
        "threshold": threshold
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("--model-path", default="outputs/ai_generated/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    result = run_inference(args.image_path, args.model_path, args.threshold)
    if result:
        print("\n--- AI-Generated Assessment ---")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
