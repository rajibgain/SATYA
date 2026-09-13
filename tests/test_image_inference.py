import os
from pathlib import Path
from PIL import Image

from src.image.detector import ImageForensicsDetector
from src.image.preprocess import preprocess_image

def test_inference_on_real_image():
    """Test the image inference pipeline on a single known 'real' test image."""
    # Find one real image from the test set
    real_test_dir = Path("data/processed/image/test/real")
    
    if not real_test_dir.exists() or not any(real_test_dir.iterdir()):
        print(f"Error: Could not find the test directory at {real_test_dir} or it is empty.")
        return

    # Grab the first image file found in the directory
    try:
        test_image_path = next(real_test_dir.glob("*.jpg"))
    except StopIteration:
        # Fallback if extensions differ
        try:
            test_image_path = next(real_test_dir.iterdir())
        except StopIteration:
            print(f"Error: Directory {real_test_dir} is empty.")
            return

    try:
        print(f"Testing Inference Pipeline...")
        print(f"Selected Image: {test_image_path.name}")
        print("-" * 30)
        
        # 2. Load with PIL
        image = Image.open(test_image_path).convert('RGB')
        
        # Preprocess the image using our custom function
        input_tensor = preprocess_image(image)
        
        # 3. Initialize detector
        detector = ImageForensicsDetector()
        
        # 4. Run prediction
        result = detector.predict(input_tensor)
        
        # 5. Print results
        print(f"Filename:   {test_image_path.name}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Device:     {result['device']}")
        
    except FileNotFoundError as fnf_err:
        print(f"Error (File Not Found): {fnf_err}")
    except RuntimeError as r_err:
        print(f"Error (Runtime): {r_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_inference_on_real_image()
