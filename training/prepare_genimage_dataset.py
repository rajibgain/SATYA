"""
FakeShield - GenImage Dataset Preparation Script

This script processes a downloaded subset of the GenImage dataset (or similar 
AI-generated image datasets) and converts it into a clean, balanced FakeShield
directory structure ready for training.

Target Structure:
data/processed/image_genimage/
    train/
        real/
        fake/
    val/ ...
    test/ ...
    dataset_manifest.json
"""

import argparse
import os
import shutil
import random
import json
import sys
from pathlib import Path

def parse_args():
    """
    Parses command-line arguments for dataset preparation.
    """
    parser = argparse.ArgumentParser(description="Prepare GenImage Dataset for FakeShield")
    parser.add_argument("--source-dir", required=True, type=str, 
                        help="Path to the downloaded GenImage subset directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/image_genimage", 
                        help="Target output directory for the FakeShield structure")
    parser.add_argument("--train-per-class", type=int, default=1000, 
                        help="Number of training images per class (real/fake)")
    parser.add_argument("--val-per-class", type=int, default=200, 
                        help="Number of validation images per class (real/fake)")
    parser.add_argument("--test-per-class", type=int, default=200, 
                        help="Number of test images per class (real/fake)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for deterministic sampling")
    parser.add_argument("--force", action="store_true", 
                        help="Force overwrite of the output directory if it already exists")
    return parser.parse_args()

def discover_files(source_dir):
    """
    Inspects the source directory to discover and classify real and fake images.
    GenImage datasets typically structure data as:
        GeneratorName/nature/img.jpg  (Real)
        GeneratorName/ai/img.jpg      (Fake)
    
    Returns lists of dictionaries containing file metadata.
    """
    real_files = []
    fake_files = []
    
    # Keywords that typically denote real or authentic images
    real_keywords = ["nature", "real", "0_real", "authentic"]
    # Keywords that typically denote AI-generated or fake images
    fake_keywords = ["ai", "fake", "1_fake", "generated"]
    
    # Walk through the entire directory tree
    for root, dirs, files in os.walk(source_dir):
        # Convert folder names to lower case for case-insensitive matching
        parent_folder = os.path.basename(root).lower()
        
        # Try to infer the generator name from the directory path
        rel_path = os.path.relpath(root, source_dir)
        parts = rel_path.split(os.sep)
        # If there are subdirectories, assume the first level is the generator/category
        generator = parts[0] if len(parts) > 1 else "unknown"
        
        for f in files:
            # Only consider standard image formats
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            file_path = os.path.join(root, f)
            
            # Check if the parent folder name indicates real or fake
            is_real = any(k in parent_folder for k in real_keywords)
            is_fake = any(k in parent_folder for k in fake_keywords)
            
            if is_real and not is_fake:
                real_files.append({"path": file_path, "generator": generator})
            elif is_fake and not is_real:
                fake_files.append({"path": file_path, "generator": generator})
            else:
                # Fallback: Check if the filename itself indicates real or fake
                is_real_file = any(k in f.lower() for k in real_keywords)
                is_fake_file = any(k in f.lower() for k in fake_keywords)
                
                if is_real_file and not is_fake_file:
                    real_files.append({"path": file_path, "generator": generator})
                elif is_fake_file and not is_real_file:
                    fake_files.append({"path": file_path, "generator": generator})
                    
    return real_files, fake_files

def process_split(files, out_dir, split, label, manifest, generators):
    """
    Copies a list of selected files into their final output directory without modifying them.
    Also records their metadata into the manifest.
    """
    count = 0
    # Create the target directory (e.g., data/processed/image_genimage/train/real)
    target_dir = out_dir / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(files):
        src_path = item["path"]
        generator = item["generator"]
        generators.add(generator)
        
        # Construct a standardized destination filename (e.g., real_00000.jpg)
        ext = os.path.splitext(src_path)[1]
        original_filename = os.path.basename(src_path)
        dest_filename = f"{label}_{i:05d}{ext}"
        dest_path = target_dir / dest_filename
        
        # Safely copy the file without resizing or altering bytes
        shutil.copy2(src_path, dest_path)
        
        # Record the file metadata in our manifest to preserve provenance
        manifest.append({
            "split": split,
            "label": label,
            "original_filename": original_filename,
            "source_path": src_path,
            "generator": generator,
            "dest_path": str(dest_path.relative_to(out_dir))
        })
        count += 1
        
    return count

def main():
    # Enforce the required security warning as specified
    print("WARNING: GenImage is intended for AI-generated-image detection and should not automatically be described as a universal deepfake detector.\n")
    
    args = parse_args()
    
    # 1. Discover and classify files
    print(f"Scanning source directory: {args.source_dir}...")
    real_files, fake_files = discover_files(args.source_dir)
    
    # 2. Validate structure discovery
    if not real_files or not fake_files:
        print("[ERROR] Could not definitively identify real and fake images based on folder or file names.")
        print("Please ensure the source directory contains folders named 'real', 'nature', 'fake', or 'ai'.")
        sys.exit(1)
        
    print(f"Found {len(real_files)} Real images and {len(fake_files)} Fake images.")
    
    # 3. Ensure we have enough data for the requested splits
    total_needed = args.train_per_class + args.val_per_class + args.test_per_class
    if len(real_files) < total_needed:
        print(f"[ERROR] Insufficient real images. Found {len(real_files)}, need {total_needed}.")
        sys.exit(1)
    if len(fake_files) < total_needed:
        print(f"[ERROR] Insufficient fake images. Found {len(fake_files)}, need {total_needed}.")
        sys.exit(1)
        
    # 4. Deterministic shuffling for reproducibility
    random.seed(args.seed)
    
    # IMPORTANT: Sort paths first to ensure order is identical across OSs before shuffling
    real_files.sort(key=lambda x: x["path"])
    fake_files.sort(key=lambda x: x["path"])
    
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    # 5. Split arrays sequentially to PREVENT DATA LEAKAGE (No overlap between splits)
    # Train
    r_train_end = args.train_per_class
    real_train = real_files[:r_train_end]
    fake_train = fake_files[:r_train_end]
    
    # Validation
    r_val_end = r_train_end + args.val_per_class
    real_val = real_files[r_train_end:r_val_end]
    fake_val = fake_files[r_train_end:r_val_end]
    
    # Test
    r_test_end = r_val_end + args.test_per_class
    real_test = real_files[r_val_end:r_test_end]
    fake_test = fake_files[r_val_end:r_test_end]
    
    # 6. Safety check for existing directories
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.force:
            print(f"\n[ERROR] Output directory '{out_dir}' already exists.")
            print("Never silently overwrite existing data. Use --force to proceed.")
            sys.exit(1)
        else:
            print(f"\n[INFO] --force supplied. Removing existing directory: {out_dir}")
            shutil.rmtree(out_dir)
            
    # Create the root output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 7. Process files and copy them to destination splits
    print("\nCopying files and generating manifest...")
    manifest = []
    generators = set()
    counts = {}
    
    # Process Real and Fake for Train
    counts["train_real"] = process_split(real_train, out_dir, "train", "real", manifest, generators)
    counts["train_fake"] = process_split(fake_train, out_dir, "train", "fake", manifest, generators)
    
    # Process Real and Fake for Validation
    counts["val_real"] = process_split(real_val, out_dir, "val", "real", manifest, generators)
    counts["val_fake"] = process_split(fake_val, out_dir, "val", "fake", manifest, generators)
    
    # Process Real and Fake for Test
    counts["test_real"] = process_split(real_test, out_dir, "test", "real", manifest, generators)
    counts["test_fake"] = process_split(fake_test, out_dir, "test", "fake", manifest, generators)
    
    # 8. Save the Dataset Manifest JSON
    manifest_path = out_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    # 9. Output Dataset Summary
    print("\n" + "=" * 50)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Train Real: {counts['train_real']}")
    print(f"Train Fake: {counts['train_fake']}")
    print(f"Val Real:   {counts['val_real']}")
    print(f"Val Fake:   {counts['val_fake']}")
    print(f"Test Real:  {counts['test_real']}")
    print(f"Test Fake:  {counts['test_fake']}")
    print("-" * 50)
    print(f"Generators Represented: {', '.join(sorted(list(generators)))}")
    print(f"Manifest saved to:      {manifest_path}")
    print("=" * 50)
    print("\n[SUCCESS] Dataset preparation complete.")

if __name__ == "__main__":
    main()
