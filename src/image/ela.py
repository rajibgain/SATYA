import io
import base64
from PIL import Image, ImageChops, ImageStat, UnidentifiedImageError

def analyze_ela(image_bytes: bytes, quality: int = 90) -> dict:
    """
    Performs Error Level Analysis (ELA) on an image by recompressing it
    and calculating the pixel differences. Returns visualization and stats.
    
    Args:
        image_bytes: Raw bytes of the uploaded image.
        quality: The JPEG compression quality level to use (default: 90).
        
    Returns:
        dict: A structured dictionary containing ELA statistics and a base64 visualization.
    """
    try:
        # 1. Open the original image
        original = Image.open(io.BytesIO(image_bytes))
        
        # Determine the format and generate an analysis note if needed
        format_name = original.format or "Unknown"
        analysis_note = ""
        if format_name.upper() != "JPEG":
            analysis_note = (
                f"Image is in {format_name} format. ELA is naturally meaningful for "
                f"lossy JPEG compression. This visualization simulates JPEG recompression, "
                f"but may not directly indicate manipulation for lossless formats."
            )
            
        # 2. Convert to RGB for consistent processing
        original = original.convert("RGB")
        
        # 3. Resize if image is extremely large to prevent expensive ELA calculations
        # and bloated base64 payloads (max dimension 1200px)
        max_dim = 1200
        if original.width > max_dim or original.height > max_dim:
            original.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
        # 4. Recompress at controlled quality
        temp_io = io.BytesIO()
        original.save(temp_io, "JPEG", quality=quality)
        temp_io.seek(0)
        recompressed = Image.open(temp_io)
        
        # 5. Calculate absolute pixel difference
        diff = ImageChops.difference(original, recompressed)
        
        # 6. Extract statistics
        stat = ImageStat.Stat(diff)
        # stat.mean is a list of means for each band (R, G, B)
        # stat.extrema is a list of tuples (min, max) for each band
        mean_error = sum(stat.mean) / len(stat.mean) if stat.mean else 0
        
        # Find the global maximum error across all channels
        extrema = diff.getextrema()
        if extrema:
            max_error = max([ex[1] for ex in extrema])
        else:
            max_error = 0
            
        # 7. Generate visualization
        # Amplify differences so they are visible by scaling max_error to 255
        if max_error > 0:
            scale = 255.0 / max_error
            # Use Image.eval or point to scale the pixel values
            diff = diff.point(lambda p: p * scale)
            
        # 8. Encode visualization to base64
        out_io = io.BytesIO()
        # Save visualization as JPEG (high quality) to return to frontend
        diff.save(out_io, format="JPEG", quality=95)
        b64_image = base64.b64encode(out_io.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64_image}"
        
        return {
            "available": True,
            "format": format_name,
            "quality": quality,
            "mean_error": round(mean_error, 2),
            "max_error": max_error,
            "analysis_note": analysis_note,
            "visualization": data_url
        }

    except Exception:
        # Silently fail for ELA rather than breaking the entire inference pipeline
        return {
            "available": False,
            "format": "Unknown",
            "quality": quality,
            "mean_error": 0,
            "max_error": 0,
            "analysis_note": "ELA could not be performed on this file.",
            "visualization": ""
        }
