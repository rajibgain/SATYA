import io
from PIL import Image, UnidentifiedImageError

def analyze_image_integrity(image_bytes: bytes) -> dict:
    """
    Performs Image Integrity and Encoding Analysis.
    Extracts actual file and decoding properties using Pillow without making
    any forensic claims about authenticity.
    
    Args:
        image_bytes: Raw bytes of the uploaded image.
        
    Returns:
        dict: A structured dictionary containing factual integrity statistics.
    """
    file_size_bytes = len(image_bytes)
    
    try:
        # Open the image to parse headers and decode properties
        img = Image.open(io.BytesIO(image_bytes))
        
        detected_format = img.format if img.format else "Unknown"
        width, height = img.size
        color_mode = img.mode
        icc_profile_present = "icc_profile" in img.info
        
        # JPEG-specific extraction
        if detected_format.upper() in ["JPEG", "MPO", "JPEG2000"]:
            # Pillow typically decodes JPEG as format "JPEG"
            is_progressive = img.info.get("progressive") or img.info.get("progression") or False
            subsampling = img.info.get("subsampling")
            
            # Subsampling might be an int in Pillow depending on version, convert to str
            if subsampling is not None:
                subsampling_str = str(subsampling)
            else:
                subsampling_str = "N/A"
                
            has_quantization = getattr(img, "quantization", None) is not None and len(img.quantization) > 0
            
            jpeg_info = {
                "available": True,
                "progressive": bool(is_progressive),
                "subsampling": subsampling_str,
                "quantization_tables": has_quantization
            }
            
            analysis_note = (
                f"Decoded as {detected_format}. JPEG encoding properties are reported below. "
                "These properties describe the file's current encoding and do not by "
                "themselves establish whether the image was edited."
            )
        else:
            jpeg_info = {
                "available": False,
                "progressive": None,
                "subsampling": None,
                "quantization_tables": None
            }
            analysis_note = (
                f"Decoded as {detected_format}. JPEG-specific encoding properties are not applicable."
            )
            
        if not icc_profile_present:
            analysis_note += " No ICC color profile was detected. Profile absence is common and is not evidence of manipulation."

        return {
            "available": True,
            "detected_format": detected_format,
            "width": width,
            "height": height,
            "color_mode": color_mode,
            "icc_profile_present": icc_profile_present,
            "file_size_bytes": file_size_bytes,
            "jpeg": jpeg_info,
            "analysis_note": analysis_note,
            "extension_consistency": None # Explicitly null since API receives bytes not filename
        }

    except UnidentifiedImageError:
        return {
            "available": False,
            "error": "Unidentified image format or corrupted byte stream."
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
