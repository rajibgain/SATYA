"""
FakeShield - Dataset Information Script

This script inspects a FakeShield image dataset and reports statistics 
such as file counts, image dimensions, formats, and dataset balance.

It expects the standard directory structure:
<data-dir>/
    train/
        real/
        fake/
    val/
        real/
        fake/
    test/
        real/
        fake/
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
from collections import Counter

def parse_args():
    """
    Parses command-line arguments for the dataset info script.
    """
    parser = argparse.ArgumentParser(description="Inspect FakeShield image dataset.")
    parser.add_argument("--data-dir", type=str, default="data/processed/image",
                        help="Path to the root of the dataset directory")
    return parser.parse_args()

def inspect_directory(dir_path: Path, max_inspect=20):
    """
    Inspects a specific directory (e.g., train/real) and returns statistics:
    total file count, formats present, sample widths, sample heights, and modes.
    """
    if not dir_path.exists():
        print(f"[ERROR] Required directory is missing: {dir_path.resolve()}")
        sys.exit(1)
        
    # Gather all files in the directory
    files = [f for f in dir_path.iterdir() if f.is_file()]
    count = len(files)
    
    formats = set()
    widths = []
    heights = []
    modes = Counter()
    
    # Inspect up to max_inspect files to gather resolution and mode stats cheaply
    inspect_count = min(count, max_inspect)
    for i in range(inspect_count):
        file_path = files[i]
        # Keep track of file extensions/formats
        formats.add(file_path.suffix.lower())
        
        try:
            # Use PIL to read the image dimensions and color mode
            with Image.open(file_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                modes[img.mode] += 1
        except Exception:
            # Silently skip files that cannot be opened as images
            pass
            
    return {
        "count": count,
        "formats": formats,
        "widths": widths,
        "heights": heights,
        "modes": modes
    }

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    print("=" * 50)
    print("FAKE SHIELD DATASET INFO")
    print("=" * 50)
    print(f"Inspecting dataset at: {data_dir.resolve()}\n")
    
    splits = ["train", "val", "test"]
    labels = ["real", "fake"]
    
    # Dictionary to hold all aggregated statistics
    stats = {}
    
    total_count = 0
    all_formats = set()
    
    # Iterate through all required directories
    for split in splits:
        stats[split] = {}
        for label in labels:
            target_dir = data_dir / split / label
            # Inspect 20 images from each of the 6 subdirectories
            dir_stats = inspect_directory(target_dir, max_inspect=20)
            stats[split][label] = dir_stats
            
            total_count += dir_stats["count"]
            all_formats.update(dir_stats["formats"])
            
    # ---------------------------------------------------------------------
    # Report Dataset Counts
    # ---------------------------------------------------------------------
    print("--- Dataset File Counts ---")
    train_real = stats["train"]["real"]["count"]
    train_fake = stats["train"]["fake"]["count"]
    val_real = stats["val"]["real"]["count"]
    val_fake = stats["val"]["fake"]["count"]
    test_real = stats["test"]["real"]["count"]
    test_fake = stats["test"]["fake"]["count"]
    
    print(f"Train Real: {train_real}")
    print(f"Train Fake: {train_fake}")
    print(f"Val Real:   {val_real}")
    print(f"Val Fake:   {val_fake}")
    print(f"Test Real:  {test_real}")
    print(f"Test Fake:  {test_fake}")
    print(f"\nTotal Images: {total_count}\n")
    
    # ---------------------------------------------------------------------
    # Report Image Formats
    # ---------------------------------------------------------------------
    print("--- Image Formats Detected ---")
    format_list = list(all_formats)
    print(f"Formats: {', '.join(format_list) if format_list else 'None'}\n")
    
    # ---------------------------------------------------------------------
    # Aggregate and Report Dimension / Mode Stats
    # ---------------------------------------------------------------------
    all_widths = []
    all_heights = []
    all_modes = Counter()
    
    for split in splits:
        for label in labels:
            all_widths.extend(stats[split][label]["widths"])
            all_heights.extend(stats[split][label]["heights"])
            all_modes.update(stats[split][label]["modes"])
            
    print("--- Sample Inspection Stats (Up to 20 images per folder) ---")
    if all_widths and all_heights:
        print(f"Minimum Width:  {min(all_widths)} px")
        print(f"Maximum Width:  {max(all_widths)} px")
        print(f"Minimum Height: {min(all_heights)} px")
        print(f"Maximum Height: {max(all_heights)} px")
    else:
        print("No valid images found to inspect dimensions.")
        
    if all_modes:
        # Format the counter output into a readable string
        common_modes = [f"{mode} ({count} samples)" for mode, count in all_modes.most_common()]
        print(f"Common Image Modes: {', '.join(common_modes)}")
    else:
        print("No valid images found to inspect modes.")
        
    # ---------------------------------------------------------------------
    # Determine Dataset Balance
    # ---------------------------------------------------------------------
    print("\n--- Dataset Balance ---")
    
    # A dataset is strictly balanced if Real == Fake exactly.
    # We will also consider it "roughly balanced" if it's within a 5% margin (45% to 55%)
    def check_balance(real_c, fake_c):
        if real_c == 0 and fake_c == 0:
            return True, True
        if real_c == fake_c:
            return True, True
            
        total = real_c + fake_c
        ratio = real_c / total
        is_roughly_balanced = (0.45 <= ratio <= 0.55)
        return False, is_roughly_balanced
        
    tr_exact, tr_rough = check_balance(train_real, train_fake)
    val_exact, val_rough = check_balance(val_real, val_fake)
    test_exact, test_rough = check_balance(test_real, test_fake)
    
    if tr_exact and val_exact and test_exact:
        print("Status: PERFECTLY BALANCED")
        print("The dataset has an exactly equal distribution of real and fake images across all splits.")
    elif tr_rough and val_rough and test_rough:
        print("Status: ROUGHLY BALANCED")
        print("The dataset has a relatively equal distribution (within 5%) of real and fake images.")
    else:
        print("Status: UNBALANCED")
        print("The dataset shows a skewed distribution between real and fake images.")
        print("Warning: Training on an unbalanced dataset may cause the model to favor the majority class.")
        
    print("=" * 50)

if __name__ == "__main__":
    main()
