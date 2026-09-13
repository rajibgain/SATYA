"""
SATYA Text Dataset Inspection Tool

Inspects a text dataset directory for:
- Directory structure validation
- File counts per split/class
- Encoding problems and empty files
- Text-length and word-count statistics
- SHA-256 exact duplicate detection (cross-split and intra-split)
- Optional metadata availability

Usage:
  python tools/inspect_text_dataset.py <dataset_path> [--output-file <path>]

Expected structure:
  dataset/
    train/
      human/
      ai_generated/
    val/
      human/
      ai_generated/
    test/
      human/
      ai_generated/

IMPORTANT:
  Exact SHA-256 hashing does NOT catch paraphrases or semantically similar text.
  Cross-split duplicates are flagged as POTENTIAL leakage, not confirmed leakage.
"""
import argparse
import os
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


SPLITS = ["train", "val", "test"]
CLASSES = ["human", "ai_generated"]


def sha256_file(filepath):
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            h.update(f.read())
    except Exception:
        return None
    return h.hexdigest()


def inspect_text_dataset(dataset_path):
    """Inspect a text dataset and return a structured report."""
    dataset_path = Path(dataset_path)

    report = {
        "dataset_path": str(dataset_path),
        "directory_structure": {},
        "file_counts": {},
        "encoding_errors": [],
        "empty_files": [],
        "text_lengths": [],
        "word_counts": [],
        "extensions_found": defaultdict(int),
        "duplicate_analysis": {
            "cross_split": [],
            "intra_split": []
        }
    }

    # Track hashes for duplicate detection: hash -> [(split, class, filepath)]
    hash_to_files = defaultdict(list)

    for split in SPLITS:
        split_dir = dataset_path / split
        report["directory_structure"][split] = "Exists" if split_dir.is_dir() else "MISSING"
        report["file_counts"][split] = {}

        for cls in CLASSES:
            cls_dir = split_dir / cls
            dir_key = f"{split}/{cls}"
            report["directory_structure"][dir_key] = "Exists" if cls_dir.is_dir() else "MISSING"

            if not cls_dir.is_dir():
                report["file_counts"][split][cls] = 0
                continue

            count = 0
            filenames = sorted(os.listdir(cls_dir))
            for fname in filenames:
                filepath = cls_dir / fname
                if not filepath.is_file():
                    continue

                ext = filepath.suffix.lower()
                report["extensions_found"][ext] += 1
                count += 1

                # Check encoding
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    report["encoding_errors"].append(str(filepath))
                    continue
                except Exception:
                    report["encoding_errors"].append(str(filepath))
                    continue

                # Check empty
                if not text.strip():
                    report["empty_files"].append(str(filepath))
                    continue

                # Stats
                report["text_lengths"].append(len(text))
                words = text.split()
                report["word_counts"].append(len(words))

                # Hashing for duplicate detection
                file_hash = sha256_file(filepath)
                if file_hash:
                    hash_to_files[file_hash].append({
                        "split": split,
                        "class": cls,
                        "file": str(filepath)
                    })

            report["file_counts"][split][cls] = count

    # Analyze duplicates
    for h, files in hash_to_files.items():
        if len(files) < 2:
            continue

        splits_involved = set(f["split"] for f in files)
        dup_entry = {
            "hash": h,
            "files": files
        }

        if len(splits_involved) > 1:
            report["duplicate_analysis"]["cross_split"].append(dup_entry)
        else:
            report["duplicate_analysis"]["intra_split"].append(dup_entry)

    return report


def print_report(report):
    """Pretty-print the inspection report."""
    print("=" * 60)
    print("TEXT DATASET INSPECTION REPORT")
    print("=" * 60)

    print("\n--- Directory Structure ---")
    for key, val in report["directory_structure"].items():
        print(f"  {key}: {val}")

    print("\n--- File Counts ---")
    for split in SPLITS:
        for cls in CLASSES:
            count = report["file_counts"].get(split, {}).get(cls, 0)
            print(f"  {split}/{cls}: {count} files")

    print("\n--- Extensions Found ---")
    print(f"  {dict(report['extensions_found'])}")

    print(f"\n--- Encoding Errors: {len(report['encoding_errors'])} ---")
    for f in report["encoding_errors"][:5]:
        print(f"  {f}")
    if len(report["encoding_errors"]) > 5:
        print(f"  ... and {len(report['encoding_errors']) - 5} more")

    print(f"\n--- Empty Files: {len(report['empty_files'])} ---")
    for f in report["empty_files"][:5]:
        print(f"  {f}")
    if len(report["empty_files"]) > 5:
        print(f"  ... and {len(report['empty_files']) - 5} more")

    print("\n--- Text-Length Statistics ---")
    lengths = report["text_lengths"]
    if lengths:
        print(f"  Count: {len(lengths)}")
        print(f"  Min: {min(lengths)} chars")
        print(f"  Max: {max(lengths)} chars")
        print(f"  Mean: {sum(lengths) / len(lengths):.1f} chars")
    else:
        print("  No valid text files found.")

    print("\n--- Word-Count Statistics ---")
    wc = report["word_counts"]
    if wc:
        print(f"  Min: {min(wc)} words")
        print(f"  Max: {max(wc)} words")
        print(f"  Mean: {sum(wc) / len(wc):.1f} words")
    else:
        print("  No valid text files found.")

    print("\n--- Duplicate Analysis (SHA-256) ---")
    cross = report["duplicate_analysis"]["cross_split"]
    intra = report["duplicate_analysis"]["intra_split"]
    print(f"  Cross-split (POTENTIAL LEAKAGE): {len(cross)} groups found")
    for group in cross[:3]:
        for f in group["files"]:
            print(f"    {f['split']}/{f['class']}: {f['file']}")
        print()
    print(f"  Intra-split: {len(intra)} groups found")

    print("\nNOTE: SHA-256 hashing detects exact duplicates only.")
    print("It does NOT catch paraphrases or semantically similar text.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SATYA Text Dataset Inspector")
    parser.add_argument("dataset_path", type=str, help="Path to text dataset root directory")
    parser.add_argument("--output-file", type=str, default=None, help="Save JSON report to file")

    args = parser.parse_args()

    print(f"Inspecting text dataset at: {args.dataset_path}")
    report = inspect_text_dataset(args.dataset_path)

    print_report(report)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
        # Convert defaultdict to dict for JSON serialization
        report["extensions_found"] = dict(report["extensions_found"])
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nReport saved to: {args.output_file}")


if __name__ == '__main__':
    main()
