import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path for absolute imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_generated.dataset import AIGeneratedDataset
from src.ai_generated.preprocess import get_train_transforms, get_eval_transforms
from src.ai_generated.model import create_model, unfreeze_stages
from src.ai_generated.evaluate import calculate_metrics, save_evaluation

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train AI-Generated Image Detector")
    parser.add_argument("--train-dir", type=str, default="dataset/train")
    parser.add_argument("--val-dir", type=str, default="dataset/val")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--output-dir", type=str, default="outputs/ai_generated")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze", type=int, default=-1, help="Number of stages to unfreeze. 0=classifier only, 4=full model")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick verification")
    
    args = parser.parse_args()
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Environment ---")
    print(f"Device: {device}")
    
    # Create dataset logic
    if args.smoke_test:
        print("Running SMOKE TEST (1 iteration, dummy data)...")
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 16
            def __getitem__(self, idx):
                return torch.randn(3, args.image_size, args.image_size), random.randint(0, 1), {}
        
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        test_dataset = DummyDataset()
    else:
        if not (Path(args.train_dir).exists() and Path(args.val_dir).exists() and Path(args.test_dir).exists()):
            print("Error: train, val, and test directories must explicitly exist.")
            return

        train_dataset = AIGeneratedDataset(args.train_dir, transform=get_train_transforms(args.image_size))
        val_dataset = AIGeneratedDataset(args.val_dir, transform=get_eval_transforms(args.image_size))
        test_dataset = AIGeneratedDataset(args.test_dir, transform=get_eval_transforms(args.image_size))
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("No data found in train/val directories! Exiting.")
            return
            
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    
    # Create model
    model = create_model(pretrained=True, freeze_backbone=args.freeze_backbone)
    if args.unfreeze >= 0:
        model = unfreeze_stages(model, args.unfreeze)
        print(f"Unfroze {args.unfreeze} stages.")
        
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    
    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    epochs = 1 if args.smoke_test else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f}")
        
        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if args.smoke_test and i == 0:
                break # Just 1 iter for smoke test
                
        scheduler.step()
        
        # Validation for Model Selection
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if args.smoke_test and i == 0:
                    break
                    
        metrics = calculate_metrics(all_labels, all_preds, all_probs)
        print(f"Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")
        roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        print(f"Val Metrics: F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | ROC-AUC: {roc_auc_str}")
        
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

    # -----------------------------
    # Final Test Evaluation
    # -----------------------------
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
        for i, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if args.smoke_test and i == 0:
                break
                
    test_metrics = calculate_metrics(all_labels, all_preds, all_probs)
    roc_auc_str = f"{test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] is not None else "N/A"
    print(f"Test Metrics: F1: {test_metrics['f1']:.4f} | Acc: {test_metrics['accuracy']:.4f} | ROC-AUC: {roc_auc_str}")
    save_evaluation(test_metrics, Path(args.output_dir) / "test_evaluation.json")

if __name__ == '__main__':
    main()
