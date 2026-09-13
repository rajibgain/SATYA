import argparse
import torch
import json
from pathlib import Path

# Add project root to path for absolute imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.audio.preprocess import AudioPreprocessor
from src.audio.model import create_model

def analyze_audio(filepath, model_path=None):
    """
    Analyzes a single audio file and returns the AI-generated probability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocessor = AudioPreprocessor()
    try:
        waveform = preprocessor.load_and_standardize(filepath)
        spectrogram = preprocessor.get_mel_spectrogram(waveform)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None
        
    model = create_model()
    if model_path and Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
        
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        spectrogram = spectrogram.unsqueeze(0).to(device)
        output = model(spectrogram)
        prob = torch.sigmoid(output).item()
        
    return prob

def main():
    parser = argparse.ArgumentParser(description="AI-Generated Audio Inference Skeleton")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--model-path", default=None, help="Path to checkpoint")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")
    
    args = parser.parse_args()
    
    prob = analyze_audio(args.audio_path, args.model_path)
    if prob is None:
        return
        
    verdict = "LIKELY FAKE" if prob >= args.threshold else "LIKELY REAL"
    status = "checkpoint-loaded" if (args.model_path and Path(args.model_path).exists()) else "untrained"
    
    result = {
        "model_name": "AudioResNet (Lightweight)",
        "processing_status": status,
        "fake_probability": round(prob, 4),
        "real_probability": round(1.0 - prob, 4),
        "verdict": verdict,
        "threshold": args.threshold
    }
    
    print("\n--- AI-Generated Audio Assessment ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    
    if status == "untrained":
        print("\nWARNING: This model is currently UNTRAINED. The resulting probabilities are random and NOT scientifically meaningful.")

if __name__ == "__main__":
    main()
