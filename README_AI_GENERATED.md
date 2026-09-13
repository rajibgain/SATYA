# SATYA: AI-Generated Image Detector (Development Skeleton)

## Purpose
This module introduces the capability to detect AI-generated images. It operates orthogonally to the existing digital image forensics pipeline, maintaining strict separation of concerns.

## Expected Dataset Structure
The dataset loader expects a directory structured as follows:
```text
dataset_root/
    real/
        image1.jpg
        ...
    ai_generated/
        image2.jpg
        ...
```
*Note: A future manifest file can provide granular group, generator, and source mapping for advanced leakage prevention.*

## Class Mapping
- `0` = Real Image
- `1` = AI-Generated Image

## Implemented Features
- **Model**: ConvNeXt-Tiny (torchvision) adapted for binary classification.
- **Dataset Loader**: Supports common formats and safely handles corruption.
- **Preprocessing**: Follows official ConvNeXt ImageNet protocols, distinct between training (augmentations) and evaluation.
- **Training Script**: Configurable parameters (epochs, lr, batch-size, freeze/unfreeze).
- **Inference Script**: Outputs probability and threshold-based assessment.
- **Evaluation**: Calculates Accuracy, Precision, Recall, F1, and extracts a confusion matrix.
- **Smoke Test**: CPU-compatible verification of the end-to-end pipeline using dummy data.

## NOT Implemented Yet
- Full dataset acquisition (ArtiFact, GenImage, etc.).
- Complete training loop execution on GPUs.
- Integration into the SATYA Flask API.
- Integration into the Unified Evidence Panel frontend.

## Commands

**Run a Smoke Test (CPU Compatible)**
```bash
python training/train_ai_generated.py --smoke-test
```

**Generate a Dataset Manifest**
```bash
python -m src.ai_generated.manifest --data-dir dataset/train --output-file manifest.json
```

**Run Inference (Once trained)**
```bash
python -m src.ai_generated.inference path/to/image.jpg --model-path outputs/ai_generated/best_model.pth
```
