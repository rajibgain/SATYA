import argparse
import os
import json
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
import torchaudio

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset(dataset_dir, output_file=None):
    dataset_dir = Path(dataset_dir)
    print(f"Inspecting audio dataset at: {dataset_dir}")
    
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    supported_exts = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
    
    report = {
        "directories_exist": {},
        "file_counts": defaultdict(dict),
        "extensions_found": Counter(),
        "invalid_corrupt_count": 0,
        "sample_rates": Counter(),
        "channels": Counter(),
        "duration_sec": [],
        "file_sizes_bytes": [],
        "exact_duplicates": {
            "cross_split": [],
            "intra_split": []
        },
        "near_duplicates": {
            "cross_split": [],
            "intra_split": []
        }
    }
    
    for split in splits:
        split_dir = dataset_dir / split
        report["directories_exist"][split] = split_dir.exists()
        for cls in classes:
            cls_dir = split_dir / cls
            exists = cls_dir.exists()
            report["directories_exist"][f"{split}/{cls}"] = exists
            if not exists:
                report["file_counts"][split][cls] = 0
                
    hash_map = {}
    size_duration_map = {}
    
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
                        
                        file_size = filepath.stat().st_size
                        report["file_sizes_bytes"].append(file_size)
                        
                        file_hash = get_file_hash(filepath)
                        if file_hash not in hash_map:
                            hash_map[file_hash] = []
                        hash_map[file_hash].append((split, str(filepath)))
                        
                        try:
                            # Load metadata to verify integrity without loading full file
                            import soundfile as sf
                            metadata = sf.info(str(filepath))
                            report["sample_rates"][metadata.samplerate] += 1
                            report["channels"][metadata.channels] += 1
                            duration = metadata.frames / metadata.samplerate
                            report["duration_sec"].append(duration)
                            
                            # Heuristic for near duplicates
                            sd_key = f"{file_size}_{duration:.3f}"
                            if sd_key not in size_duration_map:
                                size_duration_map[sd_key] = []
                            size_duration_map[sd_key].append((split, str(filepath), file_hash))
                            
                        except Exception as e:
                            report["invalid_corrupt_count"] += 1
                            
            report["file_counts"][split][cls] = count
            
    for f_hash, occurrences in hash_map.items():
        if len(occurrences) > 1:
            splits_involved = set(s for s, p in occurrences)
            if len(splits_involved) > 1:
                report["exact_duplicates"]["cross_split"].append(occurrences)
            else:
                report["exact_duplicates"]["intra_split"].append(occurrences)
                
    for key, occurrences in size_duration_map.items():
        if len(occurrences) > 1:
            # Check if they are actually just exact duplicates (same hash)
            hashes = set(h for _, _, h in occurrences)
            if len(hashes) > 1:
                # Different hashes, but same size and duration -> near duplicate
                stripped_occurrences = [(s, p) for s, p, _ in occurrences]
                splits_involved = set(s for s, p in stripped_occurrences)
                if len(splits_involved) > 1:
                    report["near_duplicates"]["cross_split"].append(stripped_occurrences)
                else:
                    report["near_duplicates"]["intra_split"].append(stripped_occurrences)
                    
    print("\n" + "="*50)
    print("AUDIO DATASET INSPECTION REPORT")
    print("="*50)
    
    print("\n--- Directory Structure ---")
    for k, v in report["directories_exist"].items():
        print(f"  {k}: {'Exists' if v else 'Missing'}")
        
    print("\n--- File Counts ---")
    for split in splits:
        for cls in classes:
            print(f"  {split}/{cls}: {report['file_counts'][split].get(cls, 0)} files")
            
    print("\n--- Audio Statistics ---")
    print(f"  Supported Extensions Found: {dict(report['extensions_found'])}")
    print(f"  Invalid/Corrupt Audio Files: {report['invalid_corrupt_count']}")
    print(f"  Sample Rates Found: {dict(report['sample_rates'])}")
    print(f"  Channel Counts Found: {dict(report['channels'])}")
    
    if report["duration_sec"]:
        durations = report["duration_sec"]
        print(f"  Durations: Min={min(durations):.2f}s, Max={max(durations):.2f}s, Mean={sum(durations)/len(durations):.2f}s")
        
    print("\n--- Duplicate Analysis ---")
    cross_exact = len(report['exact_duplicates']['cross_split'])
    intra_exact = len(report['exact_duplicates']['intra_split'])
    cross_near = len(report['near_duplicates']['cross_split'])
    intra_near = len(report['near_duplicates']['intra_split'])
    print(f"  Exact Duplicates (Cryptographic Hash):")
    print(f"    Cross-split (SEVERE LEAKAGE): {cross_exact} groups found")
    print(f"    Intra-split: {intra_exact} groups found")
    print(f"  SUSPICIOUS SIMILARITY HEURISTIC (Same size & duration, diff hash - NOT a confirmed duplicate):")
    print(f"    Cross-split: {cross_near} groups found")
    print(f"    Intra-split: {intra_near} groups found")
            
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        import copy
        json_report = copy.deepcopy(report)
        with open(output_file, 'w') as f:
            json.dump(json_report, f, indent=4)
        print(f"\nReport saved to: {output_file}")
        
    print("="*50 + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Dataset Inspector")
    parser.add_argument("dataset_dir", help="Path to the dataset root directory")
    parser.add_argument("--output-file", help="Path to save JSON report")
    
    args = parser.parse_args()
    inspect_dataset(args.dataset_dir, args.output_file)
