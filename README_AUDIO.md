# SATYA Audio Forensics Module

This is the real-data training-ready pipeline for SATYA's audio forensics module, designed to conceptually detect fake audio.

## IMPLEMENTED NOW

The current module contains the foundational engineering pipeline:
- `src/audio/preprocess.py`: Deterministic preprocessing converting audio into standardized Mel-spectrograms.
- `src/audio/features.py`: Feature extraction pipelines (waveforms, spectrograms, forensic descriptors).
- `src/audio/model.py`: Lightweight AudioResNet-style 2D CNN operating on Mel-spectrogram representations.
  - **STATUS:** Baseline architecture awaiting real-data evaluation. This computationally efficient baseline is suitable for initial real-data evaluation. Future work may compare it against stronger pretrained audio models if computationally justified.
- `src/audio/dataset.py`: PyTorch dataset traversing split audio directories (`real` and `fake`).
- `src/audio/inference.py`: CLI that returns `LIKELY FAKE` or `LIKELY REAL` verdicts based on binary probabilities, along with processing status.
- `training/train_audio.py`: Configurable training loop calculating accurate metrics including ROC-AUC and enforcing strict dataset splits.
- `tools/inspect_audio_dataset.py`: Robust CLI tool for exploring audio files, tracking durations, corrupt files, sample rates, and cryptographic leakage (duplicates).
- `tools/audio_manifest.py`: Script to generate a JSONL manifest containing file paths, labels, splits, durations, and cryptographic SHA-256 hashes.

*Note: Synthetic smoke tests have been constructed and executed to prove pipeline execution. They do NOT establish accuracy, forensic reliability, generalization, robustness, or real-world deepfake detection ability.*

## DATASET LEAKAGE

Our inspection tool distinguishes between:
- **EXACT DUPLICATE:** Exact SHA-256 cross-split duplication. This is a definite leakage finding.
- **SUSPICIOUS SIMILARITY HEURISTIC:** Files with the same duration and size, but different hashes. This is NOT a confirmed duplicate.
- **SPEAKER/SOURCE OVERLAP:** SHA-256 alone cannot detect speaker leakage or semantic near-duplicates. Provenance information is needed for robust speaker/source separation.
- **UNKNOWN:** Untested files.

## FUTURE WORK

- **Dataset Acquisition**: Large-scale audio spoofing datasets (e.g., ASVspoof, WaveFake) must be curated and preprocessed.
- **Model Training**: A robust training phase using actual acquired datasets to perform a genuine held-out forensic evaluation.
- **Frontend / API Integration**: Linking this module into the primary SATYA UI and unified evidence panel.
