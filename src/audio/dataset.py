import torch
from torch.utils.data import Dataset
from pathlib import Path
from .preprocess import AudioPreprocessor

class AudioDataset(Dataset):
    def __init__(self, root_dir=None, manifest_path=None, preprocessor=None):
        self.preprocessor = preprocessor if preprocessor else AudioPreprocessor()
        self.samples = []
        
        if manifest_path:
            import csv
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            with open(manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = 0 if row['label'] == 'real' else 1
                    # Paths in manifest are relative to PROJECT_ROOT
                    audio_path = PROJECT_ROOT / row['path']
                    self.samples.append({
                        'path': str(audio_path),
                        'label': label,
                        'utterance_id': row.get('utterance_id', '')
                    })
        elif root_dir:
            self.root_dir = Path(root_dir)
            # Load real and fake classes
            for class_name, label in [('real', 0), ('fake', 1)]:
                class_dir = self.root_dir / class_name
                if not class_dir.exists():
                    continue
                    
                all_paths = []
                for ext in ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac'):
                    all_paths.extend(class_dir.rglob(ext))
                    
                for audio_path in sorted(all_paths):
                    self.samples.append({
                        'path': str(audio_path),
                        'label': label
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['path']
        label = sample['label']
        
        try:
            # Generate standardized waveform and then mel spectrogram
            waveform = self.preprocessor.load_and_standardize(audio_path)
            spectrogram = self.preprocessor.get_mel_spectrogram(waveform)
            # Ensure it is [channels, freq, time] which is usually [1, n_mels, time]
            return spectrogram, label, {'path': audio_path}
        except Exception as e:
            # Signal that this file is corrupt/empty and should be skipped
            return None

def audio_collate_fn(batch):
    """
    Custom collate function to handle skipped corrupt files by filtering out None values.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), []
        
    spectrograms, labels, metadata = zip(*batch)
    spectrograms = torch.stack(spectrograms)
    labels = torch.tensor(labels)
    
    return spectrograms, labels, metadata
