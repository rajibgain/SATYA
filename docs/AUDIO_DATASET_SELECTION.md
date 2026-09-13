# SATYA Audio Dataset Selection

## Executive Summary
This document provides a rigorous evaluation of dataset candidates for training the SATYA Audio Forensics module (currently an `AudioResNet` baseline). The goal is to identify a dataset that supports real-world generalizability, mitigates data leakage risks, and aligns with the computational limits of our environment. Based on the analysis, **ASVspoof 2019 (Logical Access)** is recommended as the primary dataset.

## Existing Pipeline Compatibility
Our existing engineering pipeline (`src/audio/`) has been analyzed for real dataset readiness:
1. **Directory Structure:** Expects `train`, `val`, `test` splits, with `real` and `fake` subdirectories.
2. **Labels:** `0` (real/bona fide), `1` (fake/spoofed).
3. **Audio Formats Supported:** `.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`.
4. **Sample Rate Assumptions:** Automatically resampled to 16,000 Hz.
5. **Clip Duration Assumptions:** Automatically padded/truncated to exactly 5.0 seconds.
6. **Mel-spectrogram Dimensions:** `128` mel bands, resulting in a `128 x 157` tensor for 5s audio.
7. **Tensor Shape:** `[batch, 1, 128, 157]`.
8. **Variable Duration Supported:** Yes, via deterministic padding/truncation during preprocessing.
9. **Stereo Support:** Yes, automatically averaged to mono.
10. **Corrupt Files:** Gracefully skipped during dataset collation.
11. **Duplicates:** Checked via SHA-256 and size+duration heuristics.
12. **Provenance/Metadata:** Retained via `manifest.jsonl` generation.
13. **Adaptation Requirement:** Most official datasets (like ASVspoof) are distributed as flat folders with label protocols (e.g., text files). We must build an ingestion script to symlink or copy them into our `split/class` structure.

## Candidate Datasets

### A. ASVspoof 2019 (Logical Access - LA)
- **Source:** University of Edinburgh Datashare (asvspoof.org).
- **Task:** Detection of TTS and Voice Conversion (VC) attacks.
- **Composition:** ~12,148 bona fide, ~108,978 spoofed utterances.
- **Languages:** English.
- **Speakers:** 107 speakers (46 male, 61 female).
- **Synthesis Methods:** 19 different TTS/VC systems (A01-A19).
- **Partitions:** Official Train, Dev, and Eval sets (eval contains unknown attacks A07-A19 and unseen speakers).
- **Format:** FLAC, 16 kHz.
- **Duration:** ~3-10 seconds average.
- **Licensing:** ODC-By (Open Data Commons Attribution).
- **Access:** Publicly downloadable.
- **Size:** ~24 GB.

### B. WaveFake
- **Source:** Zenodo (associated with "WaveFake: A Data Set to Facilitate Audio Deepfake Detection").
- **Task:** Vocoder artifact detection.
- **Composition:** ~13,100 real (LJ Speech), ~104,885 spoofed.
- **Languages:** English.
- **Speakers:** 1 speaker (female).
- **Synthesis Methods:** 6 neural vocoders (MelGAN, Parallel WaveGAN, Multi-band MelGAN, Full-band MelGAN, HiFi-GAN, WaveGlow).
- **Partitions:** Standard partitions exist, but all share the same speaker.
- **Format:** WAV, 22.05 kHz.
- **Duration:** ~2-10 seconds.
- **Licensing:** CC BY 4.0.
- **Access:** Publicly downloadable.
- **Size:** ~16 GB.

## Leakage Risks

| Risk Type | ASVspoof 2019 (LA) | WaveFake |
| :--- | :--- | :--- |
| **Speaker Leakage** | **LOW**: Official test sets use strictly unseen speakers. | **CRITICAL**: 100% speaker overlap (single speaker dataset). |
| **Generator Leakage** | **LOW**: Eval set features unseen TTS/VC systems (A07-A19). | **HIGH**: Model may just overfit to the 6 specific vocoders. |
| **Source Leakage** | **MODERATE**: All audio originates from VCTK corpus; clean studio conditions. | **MODERATE**: Based entirely on LJ Speech studio recordings. |
| **Duplicate Leakage** | **LOW**: Cryptographically verifiable. | **LOW**: Can be verified via hashing. |
| **Class Balance** | **SKEWED**: ~10% Real, 90% Fake. Requires balancing or PR-AUC metrics. | **SKEWED**: Heavily biased towards fakes. |
| **Domain Mismatch** | **HIGH**: Clean speech won't immediately generalize to noisy phone/WhatsApp audio or non-English languages. | **EXTREME**: Will not generalize to other speakers or noisy environments. |

## Dataset Ranking

### Weighted Decision Matrix
- **Scientific Relevance (25%)**
- **Leakage Resistance (15%)**
- **Speaker Diversity (10%)**
- **Spoof/Generator Diversity (15%)**
- **Dataset Diversity (15%)**
- **Pipeline Compatibility (5%)**
- **Accessibility/Licensing (5%)**
- **Computational Practicality (5%)**
- **Suitability for Demo (5%)**

1. **ASVspoof 2019 LA**
   - **Score:** 8.5/10
   - **Strengths:** Industry standard benchmark, strict disjoint speaker/generator splits, 107 speakers.
   - **Weaknesses:** Highly imbalanced classes, clean studio audio lacks real-world noise.
   - **Recommendation:** **BEST PRIMARY DATASET**

2. **WaveFake**
   - **Score:** 5.0/10
   - **Strengths:** Excellent for spotting specific vocoder artifacts (e.g., MelGAN vs HiFi-GAN).
   - **Weaknesses:** Single speaker makes it unusable as a general deepfake detector.
   - **Recommendation:** **BEST SECONDARY/VALIDATION DATASET (for cross-dataset evaluation only)**

## Recommended Evaluation Protocol
1. **Preserve Official Splits:** For ASVspoof 2019, we MUST preserve the official train/dev/eval splits. The eval set ensures speaker-disjoint and generator-disjoint testing.
2. **Thresholding Strategy:** Optimal thresholds must be tuned strictly on the `dev` set (e.g., using Equal Error Rate - EER minimization). The `eval` set remains entirely untouched until the model is frozen.
3. **Cross-Dataset Generalization:** 
   - **Train:** ASVspoof 2019 Train
   - **Threshold:** ASVspoof 2019 Dev
   - **In-Domain Test:** ASVspoof 2019 Eval
   - **Cross-Domain Test:** WaveFake (LJ Speech). This reveals if the detector learned genuine forensic artifacts or just ASVspoof's VCTK acoustic conditions.

## Data Ingestion Design
To comply with our pipeline, the flat datasets must be converted to:
```
data/processed/audio/
├── train/
│   ├── real/
│   └── fake/
├── val/ (ASVspoof dev)
│   ├── real/
│   └── fake/
└── test/ (ASVspoof eval)
    ├── real/
    └── fake/
```
The ingestion script will read the official protocol files (`.txt` files containing file ID, speaker, system ID, and label) and generate symlinks or copies into our structure.
**Metadata Preservation:** We will generate `manifest.jsonl` maintaining original file paths, `system_id` (e.g., A01), and `speaker_id` to allow per-generator analysis later.

## Acquisition Plan
- **Target:** ASVspoof 2019 Logical Access (LA)
- **URL:** https://datashare.ed.ac.uk/handle/10283/3336
- **Access:** Direct download (zip).
- **Size:** ~24 GB.
- **Disk Space Required:** ~50 GB for extraction and dataset formatting.
- **Expected Preprocessing Time:** ~30-60 minutes.
- **Expected Training Time:** On Intel i5-12500H (CPU), 10 epochs may take 6-12 hours depending on batch size. If moved to the RX 7800 XT (ROCm/DirectML), this drops to ~30 mins.

## Post-Acquisition Audit Checklist
- [ ] dataset license verified
- [ ] source authenticity verified
- [ ] expected file count verified
- [ ] corrupted files counted
- [ ] unsupported formats counted
- [ ] duration distribution measured
- [ ] sampling-rate distribution measured
- [ ] class balance measured
- [ ] speaker distribution measured
- [ ] generator distribution measured
- [ ] exact SHA-256 duplicates checked
- [ ] cross-split exact duplicates checked
- [ ] suspicious similarity reported separately
- [ ] speaker overlap checked where metadata allows
- [ ] generator overlap checked where metadata allows
- [ ] train/val/test contamination checked
- [ ] preprocessing conversion audited
- [ ] final manifest generated
- [ ] manifest reproducible
- [ ] no test-set contamination

## First Real Experiment
- **Dataset:** ASVspoof 2019 LA
- **Model:** AudioResNet
- **Input:** Log Mel-spectrogram (128 mels, 16kHz, 5s truncation)
- **Batch Size:** 32 (Adjust for CPU memory)
- **Learning Rate:** 1e-4
- **Optimizer:** Adam
- **Epoch Range:** 10
- **Early Stopping:** Patience of 3 epochs on Validation ROC-AUC.
- **Validation Metric:** ROC-AUC
- **Production Threshold:** Calibrated via EER on the Validation set.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, Equal Error Rate (EER), and per-generator failure analysis.

## Success Criteria
- **FAILED EXPERIMENT:** ROC-AUC < 0.60 on `test`. Model failed to learn or collapsed.
- **WEAK BASELINE:** ROC-AUC 0.60 - 0.75 on `test`. Learned some features but fails on unseen generators.
- **USEFUL BASELINE:** ROC-AUC 0.75 - 0.85 on `test`. `AudioResNet` proves its viability.
- **STRONG RESULT:** ROC-AUC > 0.85 on `test` with EER < 10%. Excellent for a lightweight CNN.
- **GENERALIZABLE RESULT:** ROC-AUC > 0.75 on the *cross-dataset* test (WaveFake) without ever training on it. (Very difficult).

*Note on existing SATYA Image benchmarks: Image achieved 79.4% Accuracy and 88.3% ROC-AUC. Audio modalities face different acoustic challenges, and we do not expect Audio to perfectly mirror Image metrics immediately.*
