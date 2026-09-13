"""
SATYA Text Forensics — Training Skeleton

Train/val/test directories must be pre-separated.
This script does NOT use random_split on already-separated data.

Class mapping:
  0 = human
  1 = ai_generated (positive class)

IMPORTANT: This is a development skeleton. Do NOT use for production inference.
"""
import argparse
import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text.dataset import TextDataset
from src.text.preprocess import TextPreprocessor
from src.text.model import TextClassifierSkeleton
from src.text.evaluate import evaluate_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="SATYA AI-Generated Text Detector — Training Skeleton")
    parser.add_argument("--train-dir", type=str, default="dataset/train",
                        help="Path to training split directory (must contain human/ and ai_generated/)")
    parser.add_argument("--val-dir", type=str, default="dataset/val",
                        help="Path to validation split directory")
    parser.add_argument("--test-dir", type=str, default="dataset/test",
                        help="Path to held-out test split directory")
    parser.add_argument("--output-dir", type=str, default="outputs/text",
                        help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum token sequence length for the model")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick verification with dummy data (1 epoch, 1 batch)")

    args = parser.parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Environment ---")
    print(f"Device: {device}")

    preprocessor = TextPreprocessor()

    if args.smoke_test:
        print("Running SMOKE TEST (1 epoch, dummy data)...")

        class DummyTextDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, idx):
                # Random token IDs within vocab range
                x = torch.randint(0, 9999, (args.max_length,), dtype=torch.long)
                y = torch.tensor(random.randint(0, 1), dtype=torch.float32)
                return x, y

        train_dataset = DummyTextDataset()
        val_dataset = DummyTextDataset()
        test_dataset = DummyTextDataset()
    else:
        for d in [args.train_dir, args.val_dir, args.test_dir]:
            if not Path(d).exists():
                print(f"Error: Directory {d} does not exist.")
                return

        train_dataset = TextDataset(args.train_dir, preprocessor=preprocessor, max_seq_length=args.max_length)
        val_dataset = TextDataset(args.val_dir, preprocessor=preprocessor, max_seq_length=args.max_length)
        test_dataset = TextDataset(args.test_dir, preprocessor=preprocessor, max_seq_length=args.max_length)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("No data found in train/val directories. Exiting.")
            return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    model = TextClassifierSkeleton(vocab_size=10000, embedding_dim=128, max_length=args.max_length)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epochs = 1 if args.smoke_test else args.epochs

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if args.smoke_test and i == 0:
                break

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)

        avg_train_loss = train_loss / max(1, train_batches)
        roc_str = f"{val_metrics['roc_auc']:.4f}" if not np.isnan(val_metrics['roc_auc']) else "N/A"
        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Metrics: F1={val_metrics['f1']:.4f} | Acc={val_metrics['accuracy']:.4f} | "
              f"Precision={val_metrics['precision']:.4f} | Recall={val_metrics['recall']:.4f} | ROC-AUC={roc_str}")

        if val_metrics['f1'] > best_f1 or args.smoke_test:
            best_f1 = val_metrics['f1']
            save_path = Path(args.output_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1
            }, save_path)
            print(f"  Saved best model to {save_path}")

    # --- Final held-out test evaluation ---
    print("\n--- Final Held-out Test Evaluation ---")
    save_path = Path(args.output_dir) / "best_model.pth"
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(model, test_loader, device)
    roc_str = f"{test_metrics['roc_auc']:.4f}" if not np.isnan(test_metrics['roc_auc']) else "N/A"
    print(f"Test Metrics: F1={test_metrics['f1']:.4f} | Acc={test_metrics['accuracy']:.4f} | "
          f"Precision={test_metrics['precision']:.4f} | Recall={test_metrics['recall']:.4f} | ROC-AUC={roc_str}")

    with open(Path(args.output_dir) / "test_evaluation.json", "w") as f:
        # Convert nan to null for JSON
        serializable = {k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in test_metrics.items()}
        json.dump(serializable, f, indent=4)
    print(f"Test results saved to {Path(args.output_dir) / 'test_evaluation.json'}")


if __name__ == '__main__':
    main()
