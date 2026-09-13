import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np

class AudioPreprocessor:
    def __init__(self, 
                 target_sample_rate=16000, 
                 target_duration_sec=5.0, 
                 n_mels=128, 
                 n_fft=1024, 
                 hop_length=512):
        self.target_sample_rate = target_sample_rate
        self.target_duration_sec = target_duration_sec
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_samples = int(self.target_sample_rate * self.target_duration_sec)
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def load_and_standardize(self, filepath):
        """Loads audio, handles mono conversion, resampling, and padding/truncation."""
        try:
            waveform_np, sample_rate = sf.read(filepath, always_2d=True)
            waveform = torch.from_numpy(waveform_np).float().T
            
            # Force mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
                
            # Pad or truncate to target duration
            num_samples = waveform.shape[1]
            if num_samples > self.target_samples:
                waveform = waveform[:, :self.target_samples]
            elif num_samples < self.target_samples:
                padding = self.target_samples - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            # Normalize waveform safely
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
                
            return waveform
            
        except Exception as e:
            raise ValueError(f"Failed to load or process audio file: {str(e)}")

    def get_mel_spectrogram(self, waveform):
        """Generates a log-mel spectrogram from a standardized waveform."""
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        return log_mel_spec
