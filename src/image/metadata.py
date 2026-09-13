import io
from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata(image_bytes: bytes) -> dict:
    """
    Safely extract EXIF and general metadata from an image byte stream.
    Does NOT assert authenticity based on findings.
    """
    metadata = {
        "exif_present": False,
        "camera_make": None,
        "camera_model": None,
        "datetime": None,
        "datetime_original": None,
        "software": None,
        "orientation": None,
        "gps_present": False,
        "width": None,
        "height": None
    }
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        metadata["width"] = image.width
        metadata["height"] = image.height
        
        exif_data = image.getexif()
        
        if exif_data is not None and len(exif_data) > 0:
            metadata["exif_present"] = True
            
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "Make":
                    metadata["camera_make"] = str(value).strip()
                elif tag == "Model":
                    metadata["camera_model"] = str(value).strip()
                elif tag == "DateTime":
                    metadata["datetime"] = str(value).strip()
                elif tag == "Software":
                    metadata["software"] = str(value).strip()
                elif tag == "Orientation":
                    metadata["orientation"] = str(value).strip()
                    
            # Check DateTimeOriginal in ExifOffset IFD
            exif_ifd = exif_data.get_ifd(0x8769) # Exif IFD
            if exif_ifd:
                for tag_id, value in exif_ifd.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal":
                        metadata["datetime_original"] = str(value).strip()
                        
            # Check GPS IFD
            gps_ifd = exif_data.get_ifd(0x8825) # GPS IFD
            if gps_ifd and len(gps_ifd) > 0:
                metadata["gps_present"] = True
                
    except Exception as e:
        # Failsafe to not break the API if metadata parsing fails
        pass
        
    return metadata
