#!/usr/bin/env python3
"""Inspect a video dataset without random splitting or data leakage assumptions."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.dataset import VideoFolder
from src.video.preprocess import SUPPORTED_VIDEO_EXTENSIONS, validate_video_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_video_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def exact_duplicate_map(paths: Iterable[Path]) -> Dict[str, List[Path]]:
    sha_by_file: Dict[str, List[Path]] = defaultdict(list)
    for path in paths:
        if path.stat().st_size == 0:
            sha_by_file["ZERO_BYTE"].append(path)
            continue
        sha_by_file[sha256_file(path)].append(path)
    return {sha: value for sha, value in sha_by_file.items() if len(value) > 1}


def inspect_split(split_dir: Path, split_name: str) -> Dict[str, object]:
    total_files = list_video_files(split_dir)
    class_counts = {"real": 0, "fake": 0}
    skipped = {"zero_byte": 0, "invalid": 0, "unsupported": 0}
    metadata = []

    for path in total_files:
        label_name = path.parent.name.lower()
        if label_name in class_counts:
            class_counts[label_name] += 1
        info = validate_video_file(path)
        if not info["valid"]:
            skipped["invalid"] += 1
            if "zero-byte file" in info["reasons"]:
                skipped["zero_byte"] += 1
            continue
        metadata.append({
            "path": str(path),
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "frame_count": info["frame_count"],
            "duration_seconds": info["duration_seconds"],
            "sha256": sha256_file(path),
        })

    return {
        "split_name": split_name,
        "total_video_files": len(total_files),
        "class_counts": class_counts,
        "skipped_invalid": skipped["invalid"],
        "zero_byte_files": skipped["zero_byte"],
        "valid_videos": len(metadata),
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect a SATYA video dataset for duplicates and invalid files.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args()

    split_info = {
        split: inspect_split(directory, split)
        for split, directory in [("train", args.train_dir), ("val", args.val_dir), ("test", args.test_dir)]
    }

    all_exact_duplicates = defaultdict(list)
    all_paths = []
    for split_name, info in split_info.items():
        for record in info["metadata"]:
            all_paths.append(Path(record["path"]))
    for sha, paths in exact_duplicate_map(all_paths).items():
        all_exact_duplicates[sha] = paths

    print("SATYA VIDEO DATASET INSPECTION")
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        info = split_info[split_name]
        print(f"\n[{split_name}]")
        print(f"  total video files: {info['total_video_files']}")
        print(f"  class counts: {info['class_counts']}")
        print(f"  valid videos: {info['valid_videos']}")
        print(f"  skipped invalid: {info['skipped_invalid']}")
        print(f"  zero-byte files: {info['zero_byte_files']}")
        if info["metadata"]:
            sample = info["metadata"][0]
            print(f"  sample metadata: width={sample['width']}, height={sample['height']}, fps={sample['fps']}, frame_count={sample['frame_count']}")

    if all_exact_duplicates:
        print("\nExact duplicates detected with SHA-256:")
        for sha, paths in all_exact_duplicates.items():
            path_list = [str(path) for path in paths]
            if len({str(path.parent.parent) for path in paths}) > 1:
                print(f"  CROSS-SPLIT DUPLICATE: {sha} -> {path_list}")
            else:
                print(f"  SAME-SPLIT DUPLICATE: {sha} -> {path_list}")
    else:
        print("\nNo exact file duplicates detected.")

    # Heuristic metadata grouping is not proof of duplication.
    metadata_groups: Dict[Tuple[int, int, float, int], List[str]] = defaultdict(list)
    for split_name, info in split_info.items():
        for record in info["metadata"]:
            key = (record["width"], record["height"], record["fps"], record["frame_count"])
            metadata_groups[key].append(record["path"])
    suspicious = {key: value for key, value in metadata_groups.items() if len(value) > 1}
    if suspicious:
        print("\nHeuristic metadata collisions (not proof of duplication):")
        for key, paths in suspicious.items():
            print(f"  metadata={key} -> {paths}")


if __name__ == "__main__":
    main()
