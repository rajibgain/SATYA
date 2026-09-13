import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

# Add project root to path for absolute imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.dataset import AudioDataset, audio_collate_fn
from src.audio.preprocess import AudioPreprocessor
from src.audio.model import create_model
from src.audio.evaluate import calculate_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train AI-Generated Audio Detector Skeleton")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick verification")
    
    args = parser.parse_args()
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Environment ---")
    print(f"Device: {device}")
    
    preprocessor = AudioPreprocessor(target_sample_rate=args.sample_rate, target_duration_sec=args.duration)
    
    if args.smoke_test:
        print("Running SMOKE TEST (1 iteration, dummy data)...")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 16
            def __getitem__(self, idx):
                # Simulated Mel Spectrogram shape [1, n_mels, time]
                time_steps = int((args.sample_rate * args.duration) / preprocessor.hop_length) + 1
                return torch.randn(1, preprocessor.n_mels, time_steps), random.randint(0, 1), {}
        
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()
    else:
        if args.train_manifest and args.val_manifest:
            train_dataset = AudioDataset(manifest_path=args.train_manifest, preprocessor=preprocessor)
            val_dataset = AudioDataset(manifest_path=args.val_manifest, preprocessor=preprocessor)
            # test is optional for standard experiments
            test_dataset = []
        else:
            if not (args.train_dir and args.val_dir and Path(args.train_dir).exists() and Path(args.val_dir).exists()):
                print("Error: must provide either valid train/val dirs or train/val manifests.")
                return

            train_dataset = AudioDataset(args.train_dir, preprocessor=preprocessor)
            val_dataset = AudioDataset(args.val_dir, preprocessor=preprocessor)
            test_dataset = AudioDataset(args.test_dir, preprocessor=preprocessor) if args.test_dir else []
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("No data found in train/val datasets! Exiting.")
            return
            
        if args.limit:
            # Pick a balanced subset for training if possible
            reals = [s for s in train_dataset.samples if s['label'] == 0]
            fakes = [s for s in train_dataset.samples if s['label'] == 1]
            n_each = args.limit // 2
            train_dataset.samples = reals[:n_each] + fakes[:n_each]
            
            val_dataset.samples = val_dataset.samples[:args.limit]
            if test_dataset and len(test_dataset) > 0:
                test_dataset.samples = test_dataset.samples[:args.limit]
            
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=audio_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=audio_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=audio_collate_fn)
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    
    print('Creating model', flush=True); model = create_model()
    model.to(device)
    
    # Class imbalance handling
    if args.smoke_test:
        pos_weight = torch.tensor([1.0]).to(device)
    else:
        num_reals = sum(1 for s in train_dataset.samples if s['label'] == 0)
        num_fakes = sum(1 for s in train_dataset.samples if s['label'] == 1)
        if num_fakes > 0:
            # pos_weight scales the positive class (Fake, label=1)
            weight = num_reals / num_fakes
            print(f"Class Imbalance -> Reals: {num_reals}, Fakes: {num_fakes}, pos_weight: {weight:.4f}")
            pos_weight = torch.tensor([weight]).to(device)
        else:
            pos_weight = torch.tensor([1.0]).to(device)
            
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    epochs = 1 if args.smoke_test else args.epochs
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        print('Starting training loop', flush=True); for i, (specs, labels, _) in enumerate(train_loader):\n            print(f'Batch {i} start', flush=True)
            if specs.numel() == 0:
                continue
            specs, labels = specs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            print(f'Backward {i} start', flush=True); loss.backward()
            print(f'Optimizer {i} start', flush=True); optimizer.step()
            
            train_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  [Train] Epoch {epoch+1} - Batch {i+1}/{len(train_loader)} - Loss: {train_loss/(i+1):.4f}", flush=True)

            if args.smoke_test and i == 0:
                break
                
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (specs, labels, _) in enumerate(val_loader):
                if specs.numel() == 0:
                    continue
                specs, labels = specs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (i + 1) % 50 == 0:
                    print(f"  [Val] Epoch {epoch+1} - Batch {i+1}/{len(val_loader)}", flush=True)

                if args.smoke_test and i == 0:
                    break
                    
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")
        roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        eer_str = f"{metrics['eer']:.4f}" if metrics.get('eer') is not None else "N/A"
        print(f"Val Metrics: F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | ROC-AUC: {roc_auc_str}")
        print(f"             Balanced Acc: {metrics.get('balanced_accuracy', 0):.4f} | EER: {eer_str}")
        print(f"             Real Recall: {metrics.get('real_recall', 0):.4f} | Fake Recall: {metrics.get('fake_recall', 0):.4f}")
        if metrics.get('eer_threshold') is not None:
            print(f"             Derived EER Threshold: {metrics['eer_threshold']:.4f}")
        
        if metrics['f1'] > best_f1 or args.smoke_test:
            best_f1 = metrics['f1']
            save_path = Path(args.output_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1
            }, save_path)
            print(f"Saved best model to {save_path}")

    if len(test_dataset) > 0:
        print("\n--- Final Held-out Test Evaluation ---")
        save_path = Path(args.output_dir) / "best_model.pth"
        if save_path.exists():
            state_dict = torch.load(save_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict['model_state_dict'])
            
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for i, (specs, labels, _) in enumerate(test_loader):
                if specs.numel() == 0:
                    continue
                specs, labels = specs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(specs)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if args.smoke_test and i == 0:
                    break
                    
        test_metrics = calculate_metrics(all_labels, all_preds, all_probs)
        roc_auc_str = f"{test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] is not None else "N/A"
        eer_str = f"{test_metrics['eer']:.4f}" if test_metrics.get('eer') is not None else "N/A"
        print(f"Test Metrics: F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f} | ROC-AUC: {roc_auc_str} | EER: {eer_str}")
        
        with open(Path(args.output_dir) / "test_evaluation.json", "w") as f:
            json.dump(test_metrics, f, indent=4)

if __name__ == '__main__':
    main()
