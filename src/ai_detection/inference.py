import os
import torch
from PIL import Image
from .model import AIDetector
from .dataset import get_transforms

MODEL_PATH = "models/ai_detection/best_ai_detector.pth"

class AIImageDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = get_transforms(train=False)
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[AI Detector] Checkpoint not found at {MODEL_PATH}. Inference disabled.")
            return

        try:
            self.model = AIDetector(freeze_blocks=True)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("[AI Detector] Model loaded successfully.")
        except Exception as e:
            print(f"[AI Detector] Error loading model: {e}")
            self.model = None

    def analyze(self, image: Image.Image):
        """
        Analyzes an image and returns the probability that it is AI-generated.
        """
        if self.model is None:
            return {
                "available": False,
                "error": "Model checkpoint not found. Please train the AI detector first."
            }

        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                
            ai_prob = float(probs[1].cpu().numpy())
            
            assessment = "LIKELY AI-GENERATED" if ai_prob >= 0.50 else "LIKELY REAL"
            
            return {
                "available": True,
                "ai_probability": ai_prob,
                "assessment": assessment,
                "threshold": 0.50
            }
        except Exception as e:
            print(f"[AI Detector] Error during inference: {e}")
            return {
                "available": False,
                "error": "Inference failed."
            }

# Singleton instance
_detector = None

def analyze_ai_generation(image: Image.Image):
    global _detector
    if _detector is None:
        _detector = AIImageDetector()
    return _detector.analyze(image)
