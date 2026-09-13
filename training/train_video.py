#!/usr/bin/env python3
"""Training skeleton for SATYA video forensics.

This script intentionally trains a deliberately small CPU-friendly model on a
local synthetic or real dataset. It is a skeleton, not a production detector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.dataset import VideoFolder, build_video_datasets
from src.video.evaluate import evaluate_dataloader
from src.video.model import VideoClassifierSkeleton


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    return running_loss / max(total_samples, 1)


def save_checkpoint(model, epoch, metrics, checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train the SATYA video forensics skeleton.")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny 1-epoch CPU-only smoke configuration.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.smoke_test:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 2)
        print("Smoke-test mode enabled: running a tiny one-epoch CPU training check.")

    train_dataset = VideoFolder(
        args.train_dir,
        target_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
        normalize=True,
    )
    val_dataset = VideoFolder(
        args.val_dir,
        target_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
        normalize=True,
    )
    test_dataset = VideoFolder(
        args.test_dir,
        target_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
        normalize=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.max_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.max_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.max_workers,
        drop_last=False,
    )

    model = VideoClassifierSkeleton(num_frames=args.num_frames, image_size=args.image_size)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.5)

    metric_history = []
    best_val_f1 = float("-inf")
    best_checkpoint = PROJECT_ROOT / "models" / "video_skeleton_best.pt"
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_dataloader(model, val_loader, device, threshold=0.5)
        metric_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        })

        print(
            f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f"val_auc={val_metrics['roc_auc'] if val_metrics['roc_auc'] is not None else 'n/a'}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(model, epoch, metric_history[-1], best_checkpoint)
            print(f"Saved validation-best checkpoint: {best_checkpoint}")

        scheduler.step()

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = evaluate_dataloader(model, test_loader, device, threshold=0.5)
    print("\nHeld-out test metrics (after best validation checkpoint was selected):")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))

    metrics_path = PROJECT_ROOT / "models" / "video_training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({
            "history": metric_history,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
        }, handle, indent=2, sort_keys=True)

    print(f"Saved training history: {metrics_path}")
    print(f"Saved checkpoint: {best_checkpoint}")
    print("This is a skeleton training script; smoke-test metrics do not represent a real detector.")


if __name__ == "__main__":
    main()
