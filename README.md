# SATYA — See Beyond the Surface

SATYA is an AI-powered multimodal digital forensics engine designed to detect synthetic content, manipulations, and deepfakes across various media types.

## Purpose
The primary purpose of SATYA is to provide a robust framework for identifying AI-generated content and manipulations. It currently offers analysis capabilities for images, and is expanding into audio, video, and text.

## Current Modality Status
- **Image:** Functional. Includes AI generation detection, metadata analysis, error level analysis (ELA), and integrity checks.
- **Audio:** In development.
- **Video:** In development.
- **Text:** In development.

## Architecture Overview
SATYA provides a Flask-based API server that wraps individual inference pipelines for each modality. The frontend is a Single Page Application (SPA) that interacts with this API.
- **Inference Engines:** PyTorch-based neural networks (e.g., EfficientNet for images).
- **Backend:** Flask REST API.
- **Frontend:** Vanilla JS/HTML/CSS SPA served by Flask.

## Repository Structure
Please refer to `docs/REPOSITORY_STRUCTURE.md` for a detailed breakdown of the directories.
Important note: Code, datasets, models, and generated outputs are strictly separated.

## Environment Setup
SATYA is tested on Windows 11.

### 1. Create a Python Environment
We recommend using a virtual environment (e.g., `venv` or `conda`).
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements-common.txt
```
For a strict, reproducible CPU-only environment on Windows:
```bash
pip install -r requirements-lock-cpu.txt
```
See `ENVIRONMENT.md` for more detailed environment instructions.

## Running the Application
**Note: You do NOT need to run training to start the application.** Pre-trained models are sufficient.

```bash
python server.py
```
The server will start at `http://localhost:5000`.

## Running Tests
To run the automated test suite:
```bash
python -m pytest tests/
```

## Datasets & Models
- **Datasets:** Datasets are **NOT** included in this repository. Refer to `docs/DATASET_SETUP.md` for instructions on acquiring and structuring datasets.
- **Models:** Refer to `docs/MODEL_ARTIFACTS.md` for details on expected model checkpoints and their statuses.

## Scientific Limitations
- The current image model is a baseline and may not generalize to all deepfake methods.
- The audio model is an intermediate result and not a final general-purpose detector.
- Do not assume perfect accuracy; outputs are probabilistic indications of manipulation.
