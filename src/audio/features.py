import torch
import numpy as np
import librosa
from .preprocess import AudioPreprocessor

def extract_audio_features(filepath, preprocessor=None):
    """
    Extracts waveform, Mel-spectrogram, and basic spectral statistics.
    Returns structured data suitable for models or analysis.
    """
    if preprocessor is None:
        preprocessor = AudioPreprocessor()
        
    try:
        # 1. Standardized Waveform
        waveform = preprocessor.load_and_standardize(filepath)
        
        # 2. Mel-Spectrogram
        log_mel_spec = preprocessor.get_mel_spectrogram(waveform)
        
        # 3. Spectral Statistics
        # Basic stats over the spectrogram
        mean_spec = torch.mean(log_mel_spec).item()
        std_spec = torch.std(log_mel_spec).item()
        max_spec = torch.max(log_mel_spec).item()
        min_spec = torch.min(log_mel_spec).item()
        
        # 4. Forensic Descriptors via librosa
        wave_np = waveform.squeeze().cpu().numpy()
        sr = preprocessor.target_sample_rate
        
        zcr = librosa.feature.zero_crossing_rate(y=wave_np)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=wave_np, sr=sr)[0]
        
        return {
            "status": "success",
            "waveform": waveform,
            "spectrogram": log_mel_spec,
            "stats": {
                "mean": mean_spec,
                "std": std_spec,
                "max": max_spec,
                "min": min_spec,
                "mean_zcr": float(np.mean(zcr)),
                "mean_spectral_centroid": float(np.mean(spectral_centroid))
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
