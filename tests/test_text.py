#!/usr/bin/env python3
"""
Tests for SATYA text forensics pipeline.
"""
import sys
import tempfile
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text.inference import infer_text
from src.text.preprocess import TextPreprocessor

class TestTextPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_text_inference(self):
        text_content = "This is a test claim. Another claim is here."
        test_file = self.base_dir / "test.txt"
        test_file.write_text(text_content, encoding="utf-8")
        
        result = infer_text(str(test_file))
        
        self.assertEqual(result["verdict"], "INSUFFICIENT EVIDENCE")
        self.assertTrue(result["claims_extracted"] > 0)
        self.assertEqual(len(result["evidence_results"]), result["claims_extracted"])

    def test_text_preprocessor(self):
        text_content = "   This is some messy text. \n\n  "
        test_file = self.base_dir / "messy.txt"
        test_file.write_text(text_content, encoding="utf-8")
        
        preprocessor = TextPreprocessor()
        raw = preprocessor.validate_and_load(str(test_file))
        clean = preprocessor.preprocess(raw)
        
        self.assertEqual(clean, "This is some messy text.")

if __name__ == "__main__":
    unittest.main()
