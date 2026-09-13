"""Inference entry point for the SATYA video model skeleton."""

from typing import Any, Dict, Optional
import torch

from src.image.inference import load_model
from src.image.preprocess import preprocess_image
from .frame_extractor import extract_frames

def infer_video(
    video_path: str,
    checkpoint_path: Optional[str] = None,
    num_frames: int = 16,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run video inference by aggregating frame-level predictions."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not checkpoint_path:
        checkpoint_path = "models/image_ntire_unfrozen/best_image_model.pth"

    try:
        model = load_model(checkpoint_path, device)
        status = "checkpoint-loaded"
    except FileNotFoundError:
        model = None
        status = "untrained"
        
    # Extract frames
    frames = extract_frames(video_path, target_frames=num_frames)
    
    if status == "untrained":
        # Fake responses if model not found for structural testing
        prob_fake = 0.5
    else:
        # Preprocess frames using canonical image preprocessing
        tensors = []
        for frame in frames:
            tensors.append(preprocess_image(frame))
        
        batch_tensor = torch.cat(tensors, dim=0).to(device)
        
        # We process in chunks to prevent MKL memory constraint violation (max 16)
        chunk_size = 16
        all_fake_probs = []
        
        with torch.no_grad():
            for i in range(0, batch_tensor.size(0), chunk_size):
                chunk = batch_tensor[i:i+chunk_size]
                outputs = model(chunk)
                probs = torch.softmax(outputs, dim=1)
                # Class mapping: Real = 0, Fake = 1
                fake_probs = probs[:, 1].cpu().tolist()
                all_fake_probs.extend(fake_probs)
                
        # Aggregate heuristics: We will use the mean probability
        prob_fake = sum(all_fake_probs) / len(all_fake_probs)

    prob_real = 1.0 - prob_fake
    verdict = "LIKELY REAL" if prob_fake < threshold else "LIKELY FAKE"

    result = {
        "real_probability": float(prob_real),
        "fake_probability": float(prob_fake),
        "threshold": float(threshold),
        "verdict": verdict,
        "model_name": "EfficientNet-B0 (Video Aggregation)",
        "status": status,
    }

    if status == "untrained":
        result["warning"] = (
            "The model is untrained. Its output is not meaningful for forensic conclusions "
            "and must not be treated as evidence of manipulation."
        )

    return result
