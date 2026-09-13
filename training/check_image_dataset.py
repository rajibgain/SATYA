"""
FakeShield - Image Dataset Validation Script
Validates the downloaded image dataset before model training.
Checks for corrupted files, zero-byte files, and valid RGB image formats.
"""

import sys
from pathlib import Path
from PIL import Image

def main():
    base_dir = Path("data/processed/image")
    splits = ["train", "val", "test"]
    labels = ["real", "fake"]
    
    total_images = 0
    total_readable = 0
    total_failed = 0
    
    # Store summary information to print at the end
    summaries = []
    
    print("=" * 50)
    print("FakeShield Image Dataset Validation")
    print("=" * 50)
    print("Validating dataset... This may take a moment.")
    
    for split in splits:
        for label in labels:
            dir_path = base_dir / split / label
            
            # Record missing directories
            if not dir_path.exists():
                summaries.append({
                    "title": f"{split.upper()} {label.upper()}",
                    "files": 0,
                    "readable": 0,
                    "failed": 0,
                    "widths": [],
                    "heights": []
                })
                continue
                
            # Grab all common image file types
            files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpeg")) + list(dir_path.glob("*.webp"))
            num_files = len(files)
            
            readable_count = 0
            failed_count = 0
            widths = []
            heights = []
            
            for file_path in files:
                total_images += 1
                
                # 5. Detect zero-byte files
                if file_path.stat().st_size == 0:
                    failed_count += 1
                    total_failed += 1
                    continue
                    
                try:
                    # 1 & 2. Attempt to open and verify the image with PIL
                    with Image.open(file_path) as img:
                        img.verify() # Check for truncation or corruption
                        
                    # Re-open because verify() can limit subsequent operations on the file pointer
                    with Image.open(file_path) as img:
                        # 3. Verify it can safely be converted to RGB
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 4. Record width and height
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        
                        readable_count += 1
                        total_readable += 1
                except Exception:
                    # 7. Count failures
                    failed_count += 1
                    total_failed += 1
            
            # Save stats for the summary table
            summaries.append({
                "title": f"{split.upper()} {label.upper()}",
                "files": num_files,
                "readable": readable_count,
                "failed": failed_count,
                "widths": widths,
                "heights": heights
            })
            
    # Print the final summary table
    print("\n" + "=" * 50)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 50)
    
    for s in summaries:
        print(s["title"])
        print(f"Files: {s['files']}")
        print(f"Readable: {s['readable']}")
        print(f"Failed: {s['failed']}")
        
        # 8. Report minimum, maximum and average width/height for each split/class
        if s["readable"] > 0:
            w_min, w_max, w_avg = min(s["widths"]), max(s["widths"]), sum(s["widths"]) / s["readable"]
            h_min, h_max, h_avg = min(s["heights"]), max(s["heights"]), sum(s["heights"]) / s["readable"]
            print(f"Widths  - Min: {w_min}, Max: {w_max}, Avg: {w_avg:.1f}")
            print(f"Heights - Min: {h_min}, Max: {h_max}, Avg: {h_avg:.1f}")
        print("-" * 30)
                
    # Global counts
    print(f"Total images:   {total_images}")
    print(f"Total readable: {total_readable}")
    print(f"Total failed:   {total_failed}")
    print("=" * 50)
    
    # Exit cleanly based on success or failure
    if total_failed > 0:
        print("\n[ERROR] One or more images failed validation.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All images passed validation.")
        sys.exit(0)

if __name__ == "__main__":
    main()
