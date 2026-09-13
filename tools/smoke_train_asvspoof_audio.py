import time
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio.preprocess import AudioPreprocessor
from src.audio.model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = PROJECT_ROOT / "data" / "processed" / "audio_asvspoof2019"

class ManifestAudioDataset(Dataset):
    def __init__(self, manifest_path, limit=None):
        self.samples = []
        self.preprocessor = AudioPreprocessor()
        
        reals = []
        fakes = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = 0 if row['label'] == 'real' else 1
                item = {
                    'path': PROJECT_ROOT / row['path'],
                    'label': label,
                    'utterance_id': row['utterance_id']
                }
                if label == 0:
                    reals.append(item)
                else:
                    fakes.append(item)
                    
        if limit:
            half = limit // 2
            self.samples = reals[:half] + fakes[:limit - half]
        else:
            self.samples = reals + fakes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            waveform = self.preprocessor.load_and_standardize(str(sample['path']))
            spectrogram = self.preprocessor.get_mel_spectrogram(waveform)
            return spectrogram, sample['label'], sample['utterance_id']
        except Exception as e:
            print(f"Failed to load {sample['path']}: {e}")
            return None

def audio_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), []
    spectrograms, labels, utterance_ids = zip(*batch)
    return torch.stack(spectrograms), torch.tensor(labels, dtype=torch.float32).unsqueeze(1), utterance_ids

def main():
    print("=== ASVSPOOF REAL-DATA SMOKE TEST ===")
    start_time = time.time()
    
    train_manifest = MANIFEST_ROOT / "manifest_train.csv"
    dev_manifest = MANIFEST_ROOT / "manifest_dev.csv"
    
    print(f"\nDataset:\nASVspoof 2019 LA")
    
    train_limit = 256
    dev_limit = 64
    
    train_dataset = ManifestAudioDataset(train_manifest, limit=train_limit)
    dev_dataset = ManifestAudioDataset(dev_manifest, limit=dev_limit)
    
    print(f"\nTrain examples:\n{len(train_dataset)}")
    print(f"\nDev examples:\n{len(dev_dataset)}")
    
    real_train = sum(1 for s in train_dataset.samples if s['label'] == 0)
    fake_train = sum(1 for s in train_dataset.samples if s['label'] == 1)
    
    print(f"\nClass counts:\nTrain Real: {real_train}, Train Fake: {fake_train}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=audio_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=audio_collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    
    print(f"\nModel:\nAudioResNet")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters:\n{trainable_params:,}")
    print(f"\nDevice:\n{device}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    unreadable_files = 0
    dropped_examples = 0
    initial_loss = None
    final_loss = None
    
    nan_inf_pass = True
    opt_step_pass = False
    
    batch_shape = None
    mel_shape = None
    
    for i, (specs, labels, ids) in enumerate(train_loader):
        if len(specs) == 0:
            unreadable_files += 1
            continue
            
        dropped_examples += (16 - len(specs)) if (i < len(train_loader) - 1 and len(specs) < 16) else 0
        
        specs = specs.to(device)
        labels = labels.to(device)
        
        if batch_shape is None:
            batch_shape = list(specs.shape)
            mel_shape = list(specs.shape[1:])
            print(f"\nBatch shape:\n{batch_shape}")
            print(f"\nMel shape:\n{mel_shape}")
            
        if torch.isnan(specs).any() or torch.isinf(specs).any():
            nan_inf_pass = False
            
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        if torch.isnan(loss) or torch.isinf(loss):
            nan_inf_pass = False
            
        loss.backward()
        optimizer.step()
        
        opt_step_pass = True
        final_loss = loss.item()
        
    # Dev validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_inference_pass = False
    
    with torch.no_grad():
        for specs, labels, ids in dev_loader:
            if len(specs) == 0:
                continue
            specs = specs.to(device)
            labels = labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_batches += 1
            val_inference_pass = True
            
    avg_val_loss = val_loss / max(1, val_batches)
    
    print(f"\nInitial loss:\n{initial_loss:.4f}" if initial_loss is not None else "\nInitial loss:\nNone")
    print(f"\nFinal/last smoke loss:\n{final_loss:.4f}" if final_loss is not None else "\nFinal/last smoke loss:\nNone")
    print(f"\nValidation loss:\n{avg_val_loss:.4f}")
    
    print(f"\nValidation inference:\n{'SUCCESS' if val_inference_pass else 'FAIL'}")
    print(f"\nNaN/Inf check:\n{'PASS' if nan_inf_pass else 'FAIL'}")
    print(f"\nOptimizer step:\n{'PASS' if opt_step_pass else 'FAIL'}")
    print(f"\nUnreadable files encountered:\n{unreadable_files}")
    print(f"\nDropped examples:\n{dropped_examples}")
    
    elapsed = time.time() - start_time
    print(f"\nElapsed time:\n{elapsed:.2f} seconds")
    
    status = "PASS" if (val_inference_pass and nan_inf_pass and opt_step_pass and initial_loss is not None) else "FAIL"
    print(f"\nFINAL STATUS:\n{status}")

if __name__ == '__main__':
    main()
