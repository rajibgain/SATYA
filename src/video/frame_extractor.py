"""Video frame extraction for the SATYA skeleton."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union
from PIL import Image

from .preprocess import validate_video_file

def sample_frame_indices(total_frames: int, target_frames: int) -> List[int]:
    """Sample frames deterministically across the temporal axis."""
    if total_frames <= 0:
        return []
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if total_frames == 1:
        return [0]

    positions = np.linspace(0, total_frames - 1, num=target_frames)
    indices = np.rint(positions).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)
    return [int(idx) for idx in indices.tolist()]

def extract_frames(
    path: Union[str, Path],
    target_frames: int = 16
) -> List[Image.Image]:
    """Decode a deterministic set of frames as PIL Images without scanning the whole video."""
    file_path = Path(path)
    info = validate_video_file(file_path)
    if not info["valid"]:
        raise ValueError(f"Invalid video: {file_path} ({'; '.join(info['reasons'])})")

    total_frames = int(info["frame_count"])
    if total_frames <= 0:
        raise ValueError(f"Zero-frame video: {file_path}")

    sample_positions = sample_frame_indices(total_frames, target_frames)
    frames: List[Image.Image] = []

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video for decoding: {file_path}")

    try:
        for frame_index in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_index, total_frames - 1)))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(
                    f"Failed to decode sampled frame {frame_index} from {file_path}"
                )
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb)
            frames.append(pil_image)
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError(f"No decoded frames produced from {file_path}")
    
    return frames
