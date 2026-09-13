import os
import json
import argparse
from pathlib import Path

def generate_manifest(dataset_dir, output_file, max_samples=None):
    """
    Scans a directory and creates a dataset manifest.
    Supports future expansion for group_id, generator, etc.
    """
    dataset_dir = Path(dataset_dir)
    manifest = []
    
    class_map = {'real': 0, 'ai_generated': 1}
    
    count = 0
    for class_name, label in class_map.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
            
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
            for img_path in class_dir.rglob(ext):
                manifest.append({
                    "path": str(img_path.resolve()),
                    "label": label,
                    "source": "unknown",
                    "generator": "unknown",
                    "group_id": "unknown"
                })
                count += 1
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break
                
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest generated with {len(manifest)} entries at {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-file", default="manifest.json")
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    generate_manifest(args.data_dir, args.output_file, args.max_samples)
