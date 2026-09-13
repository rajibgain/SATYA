import os
import sys
import shutil
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

def setup_synthetic_dataset(base_dir):
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            dir_path = base_dir / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            # Create 4 files per class per split
            for i in range(4):
                file_path = dir_path / f"synth_{i}.wav"
                # Make fake files slightly different in frequency to simulate a detectable difference
                duration = 1.0
                sr = 16000
                t = np.linspace(0, duration, int(sr * duration), False)
                freq = 440 if cls == 'real' else 880
                tone = np.sin(2 * np.pi * freq * t)
                audio = tone + np.random.normal(0, 0.05, tone.shape)
                audio = audio / np.max(np.abs(audio))
                sf.write(str(file_path), audio, sr)
                
            # Create one empty/corrupt file to test robust dataloader
            corrupt_path = dir_path / "corrupt_0.wav"
            with open(corrupt_path, 'wb') as f:
                f.write(b"not a valid wav file")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    test_dir = root_dir / "dataset_audio_smoke_test"
    output_dir = root_dir / "outputs" / "audio_smoke_test"
    
    print("1. Setting up synthetic audio dataset...")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    setup_synthetic_dataset(test_dir)
    
    try:
        print("\n2. Testing Dataset Inspector...")
        subprocess.run([
            sys.executable, str(root_dir / "tools" / "inspect_audio_dataset.py"),
            str(test_dir)
        ], check=True)
        
        print("\n3. Testing Manifest Generator...")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.jsonl"
        subprocess.run([
            sys.executable, str(root_dir / "tools" / "audio_manifest.py"),
            str(test_dir), str(manifest_path)
        ], check=True)
        
        print("\n4. Testing Training Pipeline...")
        subprocess.run([
            sys.executable, str(root_dir / "training" / "train_audio.py"),
            "--train-dir", str(test_dir / "train"),
            "--val-dir", str(test_dir / "val"),
            "--test-dir", str(test_dir / "test"),
            "--output-dir", str(output_dir),
            "--epochs", "1",
            "--batch-size", "2"
        ], check=True)
        
        print("\n5. Testing Inference...")
        test_audio_file = test_dir / "test" / "fake" / "synth_0.wav"
        model_path = output_dir / "best_model.pth"
        subprocess.run([
            sys.executable, str(root_dir / "src" / "audio" / "inference.py"),
            str(test_audio_file),
            "--model-path", str(model_path)
        ], check=True)
        
        print("\n--- Audio Smoke Test Completed Successfully! ---")
        
    finally:
        print("\nCleaning up synthetic data...")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    main()
