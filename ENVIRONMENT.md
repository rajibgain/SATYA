# SATYA Environment Guide

## Supported OS
Windows 11

## Python Version
Python 3.10 or newer is recommended.

## CPU Environment
The default SATYA environment is CPU-based to ensure maximum portability and ease of setup for inference and development. If you do not require GPU acceleration for training, we recommend the CPU environment.

## PyTorch Requirements
- **torch**, **torchvision**, **torchaudio**
- When running the CPU lock file (`requirements-lock-cpu.txt`), it automatically pulls the CPU-only PyTorch wheels from the official PyTorch repository to save space and avoid CUDA driver issues on machines without NVIDIA GPUs.

## CPU vs GPU Considerations
- **Inference:** The models (like EfficientNet-B0) run fast enough on modern CPUs for single-image inference.
- **Training:** If you plan to train models, GPU acceleration (CUDA) is highly recommended. You will need to install the CUDA-enabled PyTorch binaries instead of the CPU ones.

## AMD ROCm/HIP Considerations
AMD GPU acceleration is **not** currently officially supported or verified on Windows 11 for this repository. If you have an AMD GPU, you should use the CPU environment or use WSL2 (Linux) if you wish to experiment with ROCm.

## Required Python Packages
Core dependencies include:
- `flask`, `flask-cors` (Backend)
- `torch`, `torchvision`, `torchaudio` (Deep Learning)
- `pillow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib` (Data Processing)
- `streamlit` (Alternative UI)

## FFmpeg / Audio Dependencies
For the Audio modality, `soundfile` and `torchaudio` are used. Ensure that no system-level FFmpeg conflicts exist on your Windows machine, as this can interfere with audio loading. The codebase attempts to use native implementations where possible to avoid FFmpeg DLL hell.

## Verifying the Environment
To verify your environment, you can run the test suite:
```bash
python -m pytest tests/
```

## Known Limitations
- Relying on GPU acceleration on unsupported hardware may result in fallback to CPU or driver errors.
- The repository uses `soundfile.read` to avoid Windows/FFmpeg torchaudio DLL issues; ensure standard audio formats (WAV) are used.
