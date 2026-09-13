import argparse
import os
import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

def get_file_hash(filepath):
    """Calculates SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset(dataset_dir, use_phash=False, output_file=None):
    dataset_dir = Path(dataset_dir)
    print(f"Inspecting dataset at: {dataset_dir}")
    
    splits = ['train', 'val', 'test']
    classes = ['real', 'ai_generated']
    supported_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    
    report = {
        "directories_exist": {},
        "file_counts": defaultdict(dict),
        "extensions_found": Counter(),
        "invalid_corrupt_count": 0,
        "dimensions": Counter(),
        "color_modes": Counter(),
        "file_formats": Counter(),
        "file_sizes_bytes": [],
        "exact_duplicates": {
            "cross_split": [],
            "intra_split": []
        },
        "perceptual_duplicates": {
            "cross_split": [],
            "intra_split": []
        },
        "metadata_availability": "Generator/source grouping metadata not available."
    }
    
    # 1 & 2. Directory checks
    for split in splits:
        split_dir = dataset_dir / split
        report["directories_exist"][split] = split_dir.exists()
        for cls in classes:
            cls_dir = split_dir / cls
            exists = cls_dir.exists()
            report["directories_exist"][f"{split}/{cls}"] = exists
            if not exists:
                report["file_counts"][split][cls] = 0
                
    # Metadata check
    # We look for a manifest.json or metadata.csv as a naive check. 
    # If we had a standard, we'd parse it. 
    metadata_files = list(dataset_dir.glob("*.json")) + list(dataset_dir.glob("*.csv"))
    if metadata_files:
        report["metadata_availability"] = f"Potential metadata files found: {[f.name for f in metadata_files]}"
        
    # File scanning
    hash_map = {} # sha256 -> list of (split, path)
    phash_map = {} # phash -> list of (split, path)
    
    for split in splits:
        for cls in classes:
            cls_dir = dataset_dir / split / cls
            if not cls_dir.exists():
                continue
                
            count = 0
            for filepath in cls_dir.rglob("*"):
                if filepath.is_file():
                    ext = filepath.suffix.lower()
                    if ext in supported_exts:
                        count += 1
                        report["extensions_found"][ext] += 1
                        
                        # Size stats
                        file_size = filepath.stat().st_size
                        report["file_sizes_bytes"].append(file_size)
                        
                        # Hash checks
                        file_hash = get_file_hash(filepath)
                        if file_hash not in hash_map:
                            hash_map[file_hash] = []
                        hash_map[file_hash].append((split, str(filepath)))
                        
                        # Image checks
                        try:
                            with Image.open(filepath) as img:
                                img.verify() # Verify integrity
                                
                            # Re-open to get full details and phash
                            with Image.open(filepath) as img:
                                report["dimensions"][img.size] += 1
                                report["color_modes"][img.mode] += 1
                                report["file_formats"][img.format] += 1
                                
                                if use_phash and HAS_IMAGEHASH:
                                    ph = str(imagehash.phash(img))
                                    if ph not in phash_map:
                                        phash_map[ph] = []
                                    phash_map[ph].append((split, str(filepath)))
                                    
                        except Exception as e:
                            report["invalid_corrupt_count"] += 1
                            
            report["file_counts"][split][cls] = count
            
    # Process duplicates
    for f_hash, occurrences in hash_map.items():
        if len(occurrences) > 1:
            splits_involved = set(s for s, p in occurrences)
            if len(splits_involved) > 1:
                report["exact_duplicates"]["cross_split"].append(occurrences)
            else:
                report["exact_duplicates"]["intra_split"].append(occurrences)
                
    if use_phash and HAS_IMAGEHASH:
        for p_hash, occurrences in phash_map.items():
            if len(occurrences) > 1:
                splits_involved = set(s for s, p in occurrences)
                if len(splits_involved) > 1:
                    report["perceptual_duplicates"]["cross_split"].append(occurrences)
                else:
                    report["perceptual_duplicates"]["intra_split"].append(occurrences)
                    
    # Print Report
    print("\n" + "="*50)
    print("DATASET INSPECTION REPORT")
    print("="*50)
    
    print("\n--- Directory Structure ---")
    for k, v in report["directories_exist"].items():
        print(f"  {k}: {'Exists' if v else 'Missing'}")
        
    print("\n--- File Counts ---")
    for split in splits:
        for cls in classes:
            print(f"  {split}/{cls}: {report['file_counts'][split].get(cls, 0)} files")
            
    print(f"\n--- Metadata ---")
    print(report["metadata_availability"])
            
    print("\n--- Image Statistics ---")
    print(f"  Supported Extensions Found: {dict(report['extensions_found'])}")
    print(f"  Invalid/Corrupt Images: {report['invalid_corrupt_count']}")
    
    if report["dimensions"]:
        dims = list(report["dimensions"].keys())
        print(f"  Dimensions: Min={min(dims)}, Max={max(dims)}")
        print(f"  Most Common Dimensions: {report['dimensions'].most_common(3)}")
        
    print(f"  Color Modes: {dict(report['color_modes'])}")
    print(f"  File Formats: {dict(report['file_formats'])}")
    
    if report["file_sizes_bytes"]:
        sizes = report["file_sizes_bytes"]
        print(f"  File Sizes: Min={min(sizes)}B, Max={max(sizes)}B, Mean={sum(sizes)/len(sizes):.2f}B")
        
    print("\n--- Duplicate Analysis ---")
    cross_exact = len(report['exact_duplicates']['cross_split'])
    intra_exact = len(report['exact_duplicates']['intra_split'])
    print(f"  Exact Duplicates (Cryptographic Hash):")
    print(f"    Cross-split (SEVERE LEAKAGE): {cross_exact} groups found")
    print(f"    Intra-split: {intra_exact} groups found")
    
    if use_phash:
        if HAS_IMAGEHASH:
            cross_p = len(report['perceptual_duplicates']['cross_split'])
            intra_p = len(report['perceptual_duplicates']['intra_split'])
            print(f"  Approximate Duplicates (Perceptual Hash - Not definitive):")
            print(f"    Cross-split: {cross_p} groups found")
            print(f"    Intra-split: {intra_p} groups found")
        else:
            print("  Perceptual Hash: Skipped (imagehash module not installed)")
            
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        # Sanitize report for JSON serialization (convert tuple keys to strings)
        import copy
        json_report = copy.deepcopy(report)
        json_report["dimensions"] = {str(k): v for k, v in json_report["dimensions"].items()}
        
        with open(output_file, 'w') as f:
            json.dump(json_report, f, indent=4)
        print(f"\nReport saved to: {output_file}")
        
    print("="*50 + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Dataset Inspector")
    parser.add_argument("dataset_dir", help="Path to the dataset root directory")
    parser.add_argument("--output-file", help="Path to save JSON report")
    parser.add_argument("--phash", action="store_true", help="Enable optional approximate perceptual hash check")
    
    args = parser.parse_args()
    inspect_dataset(args.dataset_dir, args.phash, args.output_file)
