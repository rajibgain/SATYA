from pathlib import Path
import csv
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROTOCOL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asvspoof2019"
    / "cm_protocols"
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
)

AUDIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asvspoof2019"
    / "audio"
    / "LA"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audio_asvspoof2019"
)

SPLITS = {
    "train": {
        "protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": "ASVspoof2019_LA_train",
    },
    "dev": {
        "protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": "ASVspoof2019_LA_dev",
    },
    "eval": {
        "protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir": "ASVspoof2019_LA_eval",
    },
}


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 without loading the whole file into RAM."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def parse_protocol(protocol_path: Path):
    records = []

    with protocol_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid protocol line at {protocol_path}:{line_number}: "
                    f"expected 5 fields, got {len(parts)}"
                )

            speaker_id, utterance_id, system_id, attack_id, original_label = parts

            if original_label not in {"bonafide", "spoof"}:
                raise ValueError(
                    f"Unknown label '{original_label}' at "
                    f"{protocol_path}:{line_number}"
                )

            label = "real" if original_label == "bonafide" else "fake"

            records.append(
                {
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "system_id": system_id,
                    "attack_id": attack_id,
                    "original_label": original_label,
                    "label": label,
                }
            )

    return records


def process_split(split_name: str, config: dict):
    protocol_path = PROTOCOL_ROOT / config["protocol"]
    audio_dir = AUDIO_ROOT / config["audio_dir"]

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol not found: {protocol_path}")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    print(f"\n=== {split_name.upper()} ===")
    print(f"Protocol: {protocol_path}")
    print(f"Audio:    {audio_dir}")

    records = parse_protocol(protocol_path)

    # Index every FLAC by utterance ID.
    audio_files = {
        path.stem: path
        for path in audio_dir.rglob("*.flac")
    }

    print(f"Protocol records: {len(records)}")
    print(f"Audio files:      {len(audio_files)}")

    missing = []
    manifest_records = []

    for record in records:
        utterance_id = record["utterance_id"]
        audio_path = audio_files.get(utterance_id)

        if audio_path is None:
            missing.append(utterance_id)
            continue

        manifest_records.append(
            {
                "split": split_name,
                **record,
                "path": str(audio_path.relative_to(PROJECT_ROOT)),
            }
        )

    extra_audio = sorted(set(audio_files) - {
        r["utterance_id"] for r in records
    })

    print(f"Manifest records: {len(manifest_records)}")
    print(f"Missing audio:    {len(missing)}")
    print(f"Unlabelled extra: {len(extra_audio)}")

    if missing:
        print("\nERROR: Missing audio:")
        for item in missing[:10]:
            print(f"  {item}")
        raise RuntimeError(
            f"{split_name}: {len(missing)} protocol entries have no audio."
        )

    if len(manifest_records) != len(records):
        raise RuntimeError(
            f"{split_name}: manifest count does not match protocol count."
        )

    # Check utterance IDs are unique.
    ids = [r["utterance_id"] for r in manifest_records]

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{split_name}: duplicate utterance IDs detected.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_ROOT / f"manifest_{split_name}.csv"

    fieldnames = [
        "split",
        "speaker_id",
        "utterance_id",
        "system_id",
        "attack_id",
        "original_label",
        "label",
        "path",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_records)

    print(f"Written: {output_path}")

    # Class counts.
    real_count = sum(r["label"] == "real" for r in manifest_records)
    fake_count = sum(r["label"] == "fake" for r in manifest_records)

    print(f"Real: {real_count}")
    print(f"Fake: {fake_count}")

    return manifest_records, extra_audio


def main():
    print("=" * 70)
    print("ASVSPOOF 2019 LA — CM PROTOCOL MANIFEST BUILDER")
    print("=" * 70)

    if not PROTOCOL_ROOT.exists():
        raise FileNotFoundError(f"Protocol root not found: {PROTOCOL_ROOT}")

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Audio root not found: {AUDIO_ROOT}")

    all_records = {}
    all_extras = {}

    for split_name, config in SPLITS.items():
        records, extras = process_split(split_name, config)
        all_records[split_name] = records
        all_extras[split_name] = extras

    # ------------------------------------------------------------
    # Cross-split leakage checks
    # ------------------------------------------------------------

    print("\n=== CROSS-SPLIT CHECKS ===")

    split_ids = {
        split: {r["utterance_id"] for r in records}
        for split, records in all_records.items()
    }

    splits = list(split_ids)

    leakage_found = False

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a = splits[i]
            b = splits[j]

            overlap = split_ids[a] & split_ids[b]

            print(f"{a} vs {b}: {len(overlap)} overlapping utterance IDs")

            if overlap:
                leakage_found = True
                print("  First overlaps:")
                for item in sorted(overlap)[:10]:
                    print(f"    {item}")

    if leakage_found:
        raise RuntimeError("Cross-split utterance leakage detected.")

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MANIFEST BUILD COMPLETE")
    print("=" * 70)

    for split, records in all_records.items():
        real_count = sum(r["label"] == "real" for r in records)
        fake_count = sum(r["label"] == "fake" for r in records)

        print(
            f"{split.upper():5s}: "
            f"{len(records):6d} total | "
            f"{real_count:6d} real | "
            f"{fake_count:6d} fake | "
            f"{len(all_extras[split]):6d} unlabelled extras"
        )

    print(f"\nOutput directory:")
    print(OUTPUT_ROOT)

    print("\nNo audio files were copied or modified.")
    print("Unlabelled archive files remain untouched.")
    print("Labels come exclusively from the official CM protocols.")


if __name__ == "__main__":
    main()
