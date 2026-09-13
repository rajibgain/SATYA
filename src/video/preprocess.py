"""Video validation for the SATYA skeleton."""

from pathlib import Path
from typing import Any, Dict, Union
import cv2

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

def validate_video_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Return a metadata dictionary and a validity flag for a file."""
    file_path = Path(path)
    info: Dict[str, Any] = {
        "path": str(file_path),
        "exists": False,
        "valid": False,
        "reasons": [],
        "name": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": 0,
        "width": None,
        "height": None,
        "frame_count": None,
        "fps": None,
        "duration_seconds": None,
    }

    if not file_path.exists():
        info["reasons"].append("file does not exist")
        return info
    if not file_path.is_file():
        info["reasons"].append("path is not a regular file")
        return info

    info["exists"] = True
    info["size_bytes"] = file_path.stat().st_size
    if info["size_bytes"] <= 0:
        info["reasons"].append("zero-byte file")

    if info["extension"] not in SUPPORTED_VIDEO_EXTENSIONS:
        info["reasons"].append(f"unsupported extension: {info['extension'] or 'missing'}")

    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            info["reasons"].append("container could not be opened")
            cap.release()
            return info

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        
        if width > 0:
            info["width"] = width
        if height > 0:
            info["height"] = height
        if frame_count > 0:
            info["frame_count"] = frame_count
        if fps > 0:
            info["fps"] = fps
            
        if frame_count > 0 and fps > 0:
            info["duration_seconds"] = max(frame_count / fps, 0.0)

        if width <= 0 or height <= 0:
            info["reasons"].append("video metadata reports zero or invalid dimensions")
        if frame_count == 0:
            info["reasons"].append("zero-frame video")
        if fps <= 0:
            info["reasons"].append("FPS could not be determined")

        ok, _ = cap.read()
        if not ok:
            info["reasons"].append("video could not be decoded")

        cap.release()
    except Exception as exc:
        info["reasons"].append(f"decode exception: {type(exc).__name__}: {exc}")
        return info

    info["valid"] = len(info["reasons"]) == 0
    return info
