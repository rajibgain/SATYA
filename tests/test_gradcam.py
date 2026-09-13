import os
from pathlib import Path
from PIL import Image
import cv2

from src.image.detector import ImageForensicsDetector
from src.image.explain import generate_explanation

def get_first_image(directory: Path) -> Path:
    for file in directory.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            return file
    raise FileNotFoundError(f"No images found in {directory}")

def main():
    try:
        # Directories
        real_dir = Path("data/processed/image/test/real")
        fake_dir = Path("data/processed/image/test/fake")
        output_dir = Path("outputs/gradcam_test")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        print("Initializing ImageForensicsDetector...")
        detector = ImageForensicsDetector()
        
        # Select images
        real_img_path = get_first_image(real_dir)
        fake_img_path = get_first_image(fake_dir)
        
        test_cases = [
            {"path": real_img_path, "true_label": "Real"},
            {"path": fake_img_path, "true_label": "Fake"}
        ]
        
        for case in test_cases:
            path = case["path"]
            true_label = case["true_label"]
            
            print(f"\nProcessing {true_label} Image: {path.name}")
            
            # Load image
            img = Image.open(path)
            
            # Run Grad-CAM
            result = generate_explanation(detector, img)
            
            pred_label = result["label"]
            confidence = result["confidence"]
            
            print(f"  True Label: {true_label}")
            print(f"  Pred Label: {pred_label}")
            print(f"  Confidence: {confidence:.4f}")
            
            # Save outputs
            prefix = true_label.lower()
            
            orig_path = output_dir / f"{prefix}_original.jpg"
            heatmap_path = output_dir / f"{prefix}_heatmap.jpg"
            overlay_path = output_dir / f"{prefix}_overlay.jpg"
            
            # Arrays are RGB, convert to BGR for cv2.imwrite
            orig_bgr = cv2.cvtColor(result["original_image"], cv2.COLOR_RGB2BGR)
            heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(orig_path), orig_bgr)
            cv2.imwrite(str(heatmap_path), heatmap_bgr)
            cv2.imwrite(str(overlay_path), overlay_bgr)
            
            # Verify dimensions and non-empty
            for p, name in [(orig_path, "Original"), (heatmap_path, "Heatmap"), (overlay_path, "Overlay")]:
                if not p.exists() or p.stat().st_size == 0:
                    raise RuntimeError(f"Failed to generate {name} image correctly at {p}")
                print(f"  Saved {name}: {p}")
                
        print("\n[SUCCESS] Grad-CAM test completed.")
        
    except Exception as e:
        print(f"\n[ERROR] Grad-CAM test failed: {e}")

if __name__ == "__main__":
    main()
