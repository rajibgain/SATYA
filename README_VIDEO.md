# SATYA Video Forensics Module

This module extends SATYA's capabilities to detect Deepfake videos.
Rather than training a completely new temporal network, it leverages the existing, validated Image Forensics module (`EfficientNet-B0`) to perform frame-level analysis.

## Architecture

1. **Preprocessing (`preprocess.py`)**: Validates the video file container and metadata (e.g. dimensions, frame counts) to reject corrupted or invalid files early.
2. **Frame Extraction (`frame_extractor.py`)**: Deterministically samples frames from the video across its duration using OpenCV (`cv2`).
3. **Inference (`inference.py`)**:
   - Routes the sampled frames into the canonical image preprocessing pipeline.
   - Computes real/fake probabilities for each frame using `EfficientNet-B0`.
   - Aggregates the probabilities to produce a final, video-level authenticity verdict.

## Notes

- This module requires the `opencv-python` package.
- It operates in a stateless manner regarding model weights—loading the exact same checkpoint used by the image module.
- Inference occurs in chunked batches (e.g. 16 frames per batch) to avoid memory access violations on CPU.
