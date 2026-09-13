# Repository Structure

This document outlines the organization of the SATYA project repository. To maintain portability, absolute paths should NOT be used within code; all paths must be relative to the project root.

## Directory Breakdown

- `src/`: The core logic of the SATYA engine.
  - `src/image/`: Logic for image analysis (inference, metadata, ELA, integrity).
  - `src/audio/`: Logic for audio analysis.
  - `src/video/`: Logic for video analysis.
  - `src/text/`: Logic for text analysis.
  - `src/ai_detection/`: Logic for AI-generation detection.
- `training/`: All scripts used to train or evaluate models. Contains routines for model definitions and hyperparameter configurations.
- `tools/`: Utility scripts, such as dataset manifest generators or audit tools.
- `tests/`: Automated unit and integration tests. Run via `pytest`.
- `frontend/`: The HTML/JS/CSS assets for the Single Page Application (SPA).
- `docs/`: Project documentation (setup, model details, architecture).
- `models/`: Destination for trained model artifacts (`.pth`, `.pt`, `.ckpt`).
- `data/`: Directory where datasets are placed locally. **Datasets are excluded from Git**.
- `outputs/`: Output folder for analysis results, confusion matrices, and training curves.
- `logs/`: Application execution and training logs.

## What Belongs in Git?
- **Code:** Python scripts, frontend assets.
- **Documentation:** Markdown files, configuration examples.
- **Small Test Assets:** Minimal mock files (under 1MB) inside `tests/` for automated testing.
- **Trained Checkpoints (via LFS):** Approved model weights used in production, tracked with Git LFS.

## What Should NOT Be in Git?
- Datasets, dataset archives (`.zip`, `.tar.gz`).
- `venv/` or `.venv/` directories.
- `__pycache__` and compiled Python files.
- Secret `.env` files.
- Developer-specific configurations or IDE files.
- System cache or temporary files.
