import argparse
import json
import hashlib
from pathlib import Path
import soundfile as sf

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_manifest(dataset_dir, output_file):
    dataset_dir = Path(dataset_dir)
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    with open(output_file, 'w') as out_f:
        for split in splits:
            for cls in classes:
                cls_dir = dataset_dir / split / cls
                if not cls_dir.exists():
                    continue
                
                label = 1 if cls == 'fake' else 0
                
                for filepath in cls_dir.rglob("*"):
                    if filepath.is_file() and filepath.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.aac'}:
                        try:
                            metadata = sf.info(str(filepath))
                            duration = metadata.frames / metadata.samplerate
                            file_hash = get_file_hash(filepath)
                            
                            entry = {
                                "filepath": str(filepath.relative_to(dataset_dir)),
                                "split": split,
                                "class": cls,
                                "label": label,
                                "duration_sec": round(duration, 3),
                                "sha256": file_hash
                            }
                            out_f.write(json.dumps(entry) + "\n")
                        except Exception as e:
                            print(f"Skipping corrupt file {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSONL Manifest for Audio Dataset")
    parser.add_argument("dataset_dir", help="Path to the dataset root directory")
    parser.add_argument("output_file", help="Path to save JSONL manifest")
    args = parser.parse_args()
    generate_manifest(args.dataset_dir, args.output_file)
    print(f"Manifest saved to {args.output_file}")
