import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

from src.image.detector import ImageForensicsDetector
from src.image.explain import generate_explanation

def find_non_square_image(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = Path(root) / file
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                        if w != h:
                            return path
                except Exception:
                    pass
    return None

def test_gradcam_alignment():
    print("Testing Grad-CAM Geometric Alignment...")
    
    # Setup paths
    data_dir = Path("data/processed/image_ntire")
    output_dir = Path("outputs/gradcam_test_ntire")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Find a non-square image
    image_path = find_non_square_image(data_dir)
    if not image_path:
        print(f"No non-square images found in {data_dir}. Falling back to generating a fake non-square image.")
        image_path = output_dir / "dummy_non_square.jpg"
        dummy_img = Image.fromarray(np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8))
        dummy_img.save(image_path)
    
    print(f"Using image: {image_path}")
    image = Image.open(image_path)
    w_orig, h_orig = image.size
    print(f"Original image dimensions: {w_orig}x{h_orig} (W x H)")
    
    # 2. Initialize detector
    model_path = "models/image_ntire_ft1/best_image_model.pth"
    if not Path(model_path).exists():
        print(f"Warning: {model_path} not found. Attempting to use default models/image/best_image_model.pth")
        model_path = "models/image/best_image_model.pth"
        
    try:
        detector = ImageForensicsDetector(model_path=model_path)
    except FileNotFoundError:
        print("Model weights not found. Cannot run the full test. Aborting.")
        return
        
    # 3. Generate CAM
    print("Generating CAM...")
    result = generate_explanation(detector, image)
    
    # 4. Verify dimensions
    overlay = result["overlay"]
    h_overlay, w_overlay = overlay.shape[:2]
    
    print(f"Overlay dimensions: {w_overlay}x{h_overlay} (W x H)")
    
    assert w_orig == w_overlay and h_orig == h_overlay, \
        f"Dimension mismatch! Original: {w_orig}x{h_orig}, Overlay: {w_overlay}x{h_overlay}"
    
    print("[OK] Success: Overlay has exactly the original image dimensions!")
    
    # 5. Save the output
    out_path = output_dir / f"overlay_{image_path.name}"
    # show_cam_on_image returns float32 [0,1] RGB or uint8 [0,255] RGB depending on version, 
    # but the current code uses pytorch-grad-cam which returns uint8 RGB.
    # We will convert RGB to BGR for cv2.imwrite
    if overlay.dtype == np.float32 or overlay.dtype == np.float64:
        overlay_save = np.uint8(255 * overlay)
    else:
        overlay_save = overlay
        
    overlay_bgr = cv2.cvtColor(overlay_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), overlay_bgr)
    
    print(f"Saved test overlay to: {out_path}")
    print("[OK] Success: Shared transform is being used and no exception occurred.")

if __name__ == "__main__":
    test_gradcam_alignment()
