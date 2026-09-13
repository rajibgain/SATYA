#!/usr/bin/env python3
"""
Tests for SATYA video forensics pipeline.
"""
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
import unittest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.preprocess import validate_video_file
from src.video.frame_extractor import extract_frames
from src.video.inference import infer_video
from src.video.evaluate import evaluate_predictions

def generate_synthetic_video(path: Path, frames=5, width=224, height=224, color=(0, 255, 0)):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path

class TestVideoPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_video_inference(self):
        test_video = self.base_dir / "test.mp4"
        generate_synthetic_video(test_video)
        
        # We need a dummy image model checkpoint since video inference uses EfficientNet-B0
        # Wait, the inference uses load_model_for_inference from image, so we can pass a None checkpoint if it loads pretrained?
        # Actually infer_video takes a path, but in the skeleton it might just use torchvision's pretrained model if checkpoint_path is None or missing.
        # Let's pass the default checkpoint path that might be expected or None.
        
        result = infer_video(
            video_path=test_video,
            checkpoint_path=None,
            num_frames=4
        )
        
        self.assertIn("verdict", result)
        self.assertIn("fake_probability", result)

    def test_video_preprocessing(self):
        test_video = self.base_dir / "test.mp4"
        generate_synthetic_video(test_video)
        
        # Valid video
        self.assertTrue(validate_video_file(test_video)["valid"])
        
        # Corrupt video
        corrupt = self.base_dir / "corrupt.mp4"
        corrupt.write_bytes(b"bad data")
        self.assertFalse(validate_video_file(corrupt)["valid"])
        
    def test_frame_extractor(self):
        test_video = self.base_dir / "test.mp4"
        generate_synthetic_video(test_video, frames=10)
        
        frames = extract_frames(test_video, target_frames=4)
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0].size, (224, 224))
        
if __name__ == "__main__":
    unittest.main()
