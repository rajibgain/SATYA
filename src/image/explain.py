import numpy as np
import torch
import cv2
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

from src.image.preprocess import get_model_transform

def generate_explanation(detector, image: Image.Image) -> dict:
    """
    Generates a Grad-CAM explanation for the model's prediction.
    
    Args:
        detector (ImageForensicsDetector): An initialized instance of the detector.
        image (PIL.Image.Image): The original input image.
        
    Returns:
        dict: A dictionary containing:
            - original_image: The original image resized to match the model input (np.ndarray RGB)
            - input_tensor: The normalized tensor used for prediction
            - predicted_class_idx: The integer class index
            - label: The string label ("Fake" or "Real")
            - confidence: The confidence score
            - heatmap: The standalone Grad-CAM heatmap (np.ndarray RGB)
            - overlay: The Grad-CAM overlay on the original image (np.ndarray RGB)
            - description: A text description to display
    """
    if not GRAD_CAM_AVAILABLE:
        raise RuntimeError("pytorch-grad-cam is not installed. Cannot generate explanation.")
        
    # 1. Preprocess the image using the shared transform
    # Convert image to RGB to ensure 3 channels
    image_rgb = image.convert("RGB")
    preprocess = get_model_transform()
    tensor = preprocess(image_rgb)
    input_tensor = tensor.unsqueeze(0)
    
    # 2. Get standard prediction. 
    pred_result = detector.predict(input_tensor)
    
    class_idx = pred_result.get("class_idx")
    if class_idx is None:
        # Fallback to existing logic: 0=Real, 1=Fake
        class_idx = 1 if pred_result["label"].lower() == "fake" else 0
        
    # 3. Setup Grad-CAM
    target_layers = [detector.model.features[-1]]
    
    input_tensor = input_tensor.to(detector.device)
    
    # Convert PIL Image to normalized float numpy array for overlay
    original_np = np.array(image_rgb)
    original_normalized = np.float32(original_np) / 255.0
    
    cam = GradCAM(model=detector.model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]
    
    try:
        # Generate the CAM mask for the predicted class
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Geometrically correct mapping of CAM coordinates back to original image
        # explicitly accounting for the Resize((224, 224)) aspect-ratio distortion.
        w, h = image.size
        
        # We explicitly map each pixel in the original (h, w) image back to the 
        # squished (224, 224) CAM space, aligning pixel centers mathematically.
        x_coords = np.arange(w, dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32)
        xv, yv = np.meshgrid(x_coords, y_coords)
        
        # Map original image coordinates to the distorted 224x224 CAM space
        xv_cam = (xv + 0.5) * (224.0 / w) - 0.5
        yv_cam = (yv + 0.5) * (224.0 / h) - 0.5
        
        # Explicitly sample the CAM at the mapped coordinates
        cam_resized = cv2.remap(grayscale_cam, xv_cam, yv_cam, interpolation=cv2.INTER_LINEAR)
        
        # Overlay on the original image
        overlay = show_cam_on_image(original_normalized, cam_resized, use_rgb=True, image_weight=0.6)
        
        # Heatmap only 
        heatmap_uint8 = np.uint8(255 * grayscale_cam)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        return {
            "original_image": original_np,
            "input_tensor": input_tensor.cpu(),
            "predicted_class_idx": class_idx,
            "label": pred_result["label"],
            "confidence": pred_result["confidence"],
            "heatmap": heatmap_rgb,
            "overlay": overlay,
            "description": "Grad-CAM shows model activation/attention mapped correctly to the original aspect ratio."
        }
    except Exception as e:
        raise RuntimeError(f"Failed to generate Grad-CAM: {e}")
