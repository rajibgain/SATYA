from pathlib import Path
import csv
from collections import Counter
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audio_asvspoof2019"
)

SPLITS = ["train", "dev", "eval"]


def audit_split(split):
    manifest_path = MANIFEST_ROOT / f"manifest_{split}.csv"

    print(f"\n{'=' * 70}")
    print(f"{split.upper()} AUDIO INTEGRITY AUDIT")
    print(f"{'=' * 70}")

    sample_rates = Counter()
    channels = Counter()

    durations = []

    readable = 0
    unreadable = 0

    unreadable_examples = []

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(reader, start=1):
            path = PROJECT_ROOT / row["path"]

            try:
                info = sf.info(str(path))

                readable += 1
                sample_rates[info.samplerate] += 1
                channels[info.channels] += 1
                durations.append(info.duration)

            except Exception as exc:
                unreadable += 1

                if len(unreadable_examples) < 10:
                    unreadable_examples.append(
                        (row["utterance_id"], str(path), str(exc))
                    )

    print(f"Manifest records : {readable + unreadable:,}")
    print(f"Readable         : {readable:,}")
    print(f"Unreadable       : {unreadable:,}")

    print("\nSample rates:")
    for rate, count in sorted(sample_rates.items()):
        print(f"  {rate} Hz : {count:,}")

    print("\nChannels:")
    for channel_count, count in sorted(channels.items()):
        print(f"  {channel_count} channel(s) : {count:,}")

    if durations:
        print("\nDuration statistics:")
        print(f"  Minimum : {min(durations):.3f} sec")
        print(f"  Maximum : {max(durations):.3f} sec")
        print(f"  Mean    : {sum(durations) / len(durations):.3f} sec")

        sorted_durations = sorted(durations)

        def percentile(p):
            index = int((len(sorted_durations) - 1) * p)
            return sorted_durations[index]

        print(f"  P50     : {percentile(0.50):.3f} sec")
        print(f"  P95     : {percentile(0.95):.3f} sec")
        print(f"  P99     : {percentile(0.99):.3f} sec")

        longer_than_5 = sum(d > 5.0 for d in durations)
        shorter_than_5 = sum(d < 5.0 for d in durations)

        print(f"\nDuration relative to 5-second model window:")
        print(f"  < 5 sec : {shorter_than_5:,}")
        print(f"  > 5 sec : {longer_than_5:,}")
        print(f"  = 5 sec : {len(durations) - shorter_than_5 - longer_than_5:,}")

    if unreadable_examples:
        print("\nFirst unreadable files:")

        for utterance_id, path, error in unreadable_examples:
            print(f"\n  ID: {utterance_id}")
            print(f"  Path: {path}")
            print(f"  Error: {error}")


def main():
    print("=" * 70)
    print("ASVSPOOF 2019 LA — AUDIO INTEGRITY AUDIT")
    print("=" * 70)

    for split in SPLITS:
        audit_split(split)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
