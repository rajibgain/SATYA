import io
import base64
import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image, UnidentifiedImageError

def to_base64(img_array):
    """Encode numpy array as a base64 PNG data URL."""
    _, buffer = cv2.imencode('.png', img_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def analyze_image_artifacts(image_bytes: bytes) -> dict:
    """
    Independent analytical visualization layer.
    Extracts numerical signals: Noise Residual, Local Variation, Edge Density,
    JPEG block periodicity, and Frequency Domain representation.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        is_jpeg = img.format and img.format.upper() in ["JPEG", "MPO", "JPEG2000"]
        
        # Convert to RGB then Grayscale
        img_rgb = img.convert('RGB')
        # Safely resize to a max dimension for performance while preserving aspect ratio
        MAX_DIM = 1024
        if max(img_rgb.size) > MAX_DIM:
            img_rgb.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        
        # Convert to OpenCV format (numpy array)
        img_np = np.array(img_rgb)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32)

        # 1. Noise Residual
        blurred = cv2.GaussianBlur(gray_f, (3, 3), 1.0)
        residual = gray_f - blurred
        mean_abs_res = float(np.mean(np.abs(residual)))
        max_abs_res = float(np.max(np.abs(residual)))
        
        # Shift zero to 128 for visualization
        res_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)

        # 2. Local Variation (Standard Deviation via Uniform Filter)
        c1 = uniform_filter(gray_f, size=3)
        c2 = uniform_filter(gray_f**2, size=3)
        variance = np.maximum(c2 - c1**2, 0)
        std_dev = np.sqrt(variance)
        
        var_mean = float(np.mean(std_dev))
        var_median = float(np.median(std_dev))
        var_max = float(np.max(std_dev))
        var_vis = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 3. Edge Density (Sobel)
        sx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        edge_threshold = 30 # Out of 255
        density = float(np.sum(mag_norm > edge_threshold)) / mag_norm.size
        edge_vis = mag_norm.astype(np.uint8)

        # 4. JPEG 8x8 Block Analysis
        jpeg_analysis = {"available": False}
        if is_jpeg:
            h, w = gray.shape
            h_adj = (h // 8) * 8
            w_adj = (w // 8) * 8
            
            if h_adj >= 8 and w_adj >= 8:
                gray_adj = gray_f[:h_adj, :w_adj]
                diff_h = np.abs(gray_adj[:, :-1] - gray_adj[:, 1:])
                diff_v = np.abs(gray_adj[:-1, :] - gray_adj[1:, :])
                
                pad_h = np.zeros((h_adj, w_adj), dtype=np.float32)
                pad_h[:, :-1] = diff_h
                pad_v = np.zeros((h_adj, w_adj), dtype=np.float32)
                pad_v[:-1, :] = diff_v
                
                total_diff = pad_h + pad_v
                blocks_diff = total_diff.reshape(h_adj // 8, 8, w_adj // 8, 8).transpose(0, 2, 1, 3)
                avg_block = np.mean(blocks_diff, axis=(0, 1))
                
                # Scale the 8x8 matrix to 256x256 using nearest neighbor to visualize the grid
                jpeg_vis = cv2.resize(avg_block, (256, 256), interpolation=cv2.INTER_NEAREST)
                jpeg_vis = cv2.normalize(jpeg_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                jpeg_analysis = {
                    "available": True,
                    "visualization": to_base64(jpeg_vis)
                }

        # 5. Frequency Domain (FFT)
        fft_shifted = np.fft.fftshift(np.fft.fft2(gray_f))
        magnitude = np.log(np.abs(fft_shifted) + 1)
        freq_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {
            "available": True,
            "noise_residual": {
                "available": True,
                "visualization": to_base64(res_vis),
                "mean_absolute_residual": float(np.mean(np.abs(residual))),
                "max_absolute_residual": float(np.max(np.abs(residual)))
            },
            "local_variation": {
                "available": True,
                "visualization": to_base64(var_vis),
                "mean": float(np.mean(std_dev)),
                "max": float(np.max(std_dev))
            },
            "edge_analysis": {
                "available": True,
                "visualization": to_base64(edge_vis),
                "edge_density": float(density)
            },
            "jpeg_block_analysis": jpeg_analysis,
            "frequency_analysis": {
                "available": True,
                "visualization": to_base64(freq_vis)
            }
        }

    except UnidentifiedImageError:
        return {
            "available": False,
            "error": "Unidentified image format."
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "available": False,
            "error": str(e) or repr(e)
        }
