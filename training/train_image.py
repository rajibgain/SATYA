"""
FakeShield - Image Forensics Training Pipeline
Trains an EfficientNet-B0 model to distinguish between real and fake images.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# 17. Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2. Training classes must be: real = 0, fake = 1.
class FakeShieldImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):
        # Enforce mapping: real = 0, fake = 1
        classes = ["real", "fake"]
        class_to_idx = {"real": 0, "fake": 1}
        return classes, class_to_idx

import sys
from pathlib import Path

# Add project root to sys.path before importing src modules
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.image.preprocess import get_model_transform

def get_transforms():
    """
    Returns the training, validation, and test transforms.
    4. Training preprocessing must be EXACTLY the same as validation/test preprocessing.
    """
    # 5. Use exactly the canonical preprocessing
    # 6. Do NOT use random resizing/cropping that differs between train and validation/test.
    transform = get_model_transform()
    
    return transform, transform

def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], label='Train F1 (Approx)')
    plt.plot(epochs, history['val_f1'], label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, dataloader, criterion, device, use_amp, return_probs=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 16. Use mixed precision ONLY when CUDA is available.
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # Probabilities for class 1 (fake)
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0 # Handle case where only one class is present
    if return_probs:
        return epoch_loss, acc, prec, rec, f1, roc_auc, all_labels, all_preds, all_probs
    return epoch_loss, acc, prec, rec, f1, roc_auc, all_labels, all_preds

def train(args):
    set_seed(args.seed)
    
    # ---------------------------------------------------------
    # 1. Check directories (Requirement 31 & 33)
    # ---------------------------------------------------------
    base_dir = Path(args.data_dir)
    print(f"Using dataset directory: {base_dir}")
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"[ERROR] Directory {d} does not exist. The expected structure is <data-dir>/{{train,val,test}}/{{real,fake}}.")
            sys.exit(1)
            
    # ---------------------------------------------------------
    # 2. Setup datasets and dataloaders
    # ---------------------------------------------------------
    train_transform, val_test_transform = get_transforms()
    
    # 1. Use ImageFolder for the six class directories.
    train_dataset = FakeShieldImageFolder(train_dir, transform=train_transform)
    val_dataset = FakeShieldImageFolder(val_dir, transform=val_test_transform)
    test_dataset = FakeShieldImageFolder(test_dir, transform=val_test_transform)
    
    print("\n" + "=" * 50)
    print("FAKE SHIELD IMAGE FORENSICS TRAINING")
    print("=" * 50)
    # 30. Add a clear warning before training:
    if args.data_dir == "data/processed/image":
        print("WARNING: CIFAKE contains 32x32 images and this experiment is a baseline, not a universal deepfake detector.")
    else:
        print("WARNING: Training on a custom dataset. Ensure it meets the expected directory structure and image requirements.")
    print("-" * 50)
    
    # 2. Ensure the mapping is printed and verified.
    print(f"Class mapping: {train_dataset.class_to_idx}")
    assert train_dataset.class_to_idx == {"real": 0, "fake": 1}, "Class mapping is incorrect!"
    
    # 27. Print total number of train/validation/test images before training.
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---------------------------------------------------------
    # 3. Setup Model
    # ---------------------------------------------------------
    # 15. Automatically select CUDA if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load ImageNet pretrained weights for transfer learning
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # 12. Freeze the EfficientNet backbone initially.
    for param in model.features.parameters():
        param.requires_grad = False
        
    num_blocks = len(model.features)
    
    if args.unfreeze > 0:
        unfreeze_start = max(0, num_blocks - args.unfreeze)
        print(f"Unfreezing the final {args.unfreeze} feature blocks (from block {unfreeze_start} to {num_blocks-1}).")
        for i in range(unfreeze_start, num_blocks):
            for param in model.features[i].parameters():
                param.requires_grad = True
    else:
        print("Freezing the EfficientNet backbone. Training classifier head only.")
        
    # 13. Train the classifier head first. Replace classifier head for 2 classes (Real vs Fake)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Ensure classifier is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # Add a safety check that prints trainable vs total params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    
    # ---------------------------------------------------------
    # 4. Training Setup
    # ---------------------------------------------------------
    # 8. Use an equivalent class-weighting strategy
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (2.0 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nUsing class-weighting strategy in CrossEntropyLoss.")
    print(f"Class counts: Real={class_counts[0]}, Fake={class_counts[1]}")
    print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # 10. Use AdamW.
    # Use discriminative learning rates: newly unfrozen backbone = 1e-5, classifier = args.lr
    if args.unfreeze > 0:
        classifier_params = []
        for param in model.classifier.parameters():
            classifier_params.append(param)
            
        backbone_params = []
        unfreeze_start = max(0, num_blocks - args.unfreeze)
        for i in range(unfreeze_start, num_blocks):
            for param in model.features[i].parameters():
                backbone_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': classifier_params, 'lr': args.lr}
        ])
    else:
        optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr)
        
    # 11. Use a learning-rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    use_amp = (device.type == 'cuda')
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        
    # Directories for outputs
    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 20. Save checkpoint to models/image/best_image_model.pth
    best_model_path = models_dir / "best_image_model.pth"
    checkpoint_path = models_dir / "last_checkpoint.pth"
    best_val_f1 = 0.0
    start_epoch = 0
    
    history = {
        'train_loss': [], 'train_f1': [],
        'val_loss': [], 'val_f1': [],
        'val_acc': [], 'val_prec': [], 'val_rec': [], 'val_roc_auc': []
    }
    
    if args.resume:
        if checkpoint_path.exists():
            print(f"\nResuming training from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_f1 = checkpoint['best_val_f1']
            history = checkpoint.get('history', history)
            
            # Load RNG states
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
            if 'torch_cuda_rng_state' in checkpoint and checkpoint['torch_cuda_rng_state'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['torch_cuda_rng_state']])
            if 'np_rng_state' in checkpoint:
                np.random.set_state(checkpoint['np_rng_state'])
            if 'random_rng_state' in checkpoint:
                random.setstate(checkpoint['random_rng_state'])
                
            print(f"Resumed at epoch {start_epoch} with Best Val F1: {best_val_f1:.4f}")
        else:
            print(f"\n[WARNING] --resume specified but {checkpoint_path} not found. Starting from scratch.")
    
    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    if args.analyze_threshold:
        print("\n" + "=" * 50)
        print("RUNNING THRESHOLD ANALYSIS ON VALIDATION SET")
        print("=" * 50)
        
        if not best_model_path.exists():
            print(f"[ERROR] Best model not found at {best_model_path}")
            sys.exit(1)
            
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        
        print("Evaluating on validation set to collect probabilities...")
        val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc, val_labels, _, val_probs = evaluate_model(
            model, val_loader, criterion, device, use_amp, return_probs=True
        )
        
        print("\n--- NORMAL EVALUATE_MODEL() METRICS ---")
        print(f"Validation F1:        {val_f1:.4f}")
        print(f"Validation Accuracy:  {val_acc:.4f}")
        print(f"Validation Precision: {val_prec:.4f}")
        print(f"Validation Recall:    {val_rec:.4f}")
        print(f"Validation ROC-AUC:   {val_roc_auc:.4f}")
        
        import csv
        results = []
        best_f1 = 0
        best_thresh = 0.5
        best_metrics = {}
        metrics_at_050 = None
        
        # Test thresholds from 0.10 to 0.90
        thresholds = np.arange(0.10, 0.91, 0.01)
        for thresh in thresholds:
            preds = (np.array(val_probs) >= thresh).astype(int)
            acc = accuracy_score(val_labels, preds)
            prec = precision_score(val_labels, preds, zero_division=0)
            rec = recall_score(val_labels, preds, zero_division=0)
            f1 = f1_score(val_labels, preds, zero_division=0)
            
            cm = confusion_matrix(val_labels, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
                
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            metrics = {
                "threshold": float(thresh),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "fpr": float(fpr),
                "fnr": float(fnr)
            }
            results.append(metrics)
            
            if round(thresh, 2) == 0.50:
                metrics_at_050 = metrics
                
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = metrics
                
        # Save JSON
        with open(outputs_dir / "image_threshold_analysis.json", "w") as f:
            json.dump({
                "best_threshold": best_thresh,
                "best_metrics": best_metrics,
                "all_results": results
            }, f, indent=4)
            
        # Save CSV
        with open(outputs_dir / "image_threshold_analysis.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "accuracy", "precision", "recall", "f1_score", "fpr", "fnr"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\n--- THRESHOLD ANALYSIS RESULTS ---")
        if metrics_at_050:
            print(f"--- METRICS AT THRESHOLD 0.50 ---")
            print(f"Validation F1:        {metrics_at_050['f1_score']:.4f}")
            print(f"Validation Accuracy:  {metrics_at_050['accuracy']:.4f}")
            print(f"Validation Precision: {metrics_at_050['precision']:.4f}")
            print(f"Validation Recall:    {metrics_at_050['recall']:.4f}")
            print(f"FPR: {metrics_at_050['fpr']:.4f} | FNR: {metrics_at_050['fnr']:.4f}\n")
            
        print(f"--- BEST THRESHOLD ({best_thresh:.2f}) ---")
        print(f"Validation F1:        {best_f1:.4f}")
        print(f"Validation Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"Validation Precision: {best_metrics['precision']:.4f}")
        print(f"Validation Recall:    {best_metrics['recall']:.4f}")
        print(f"FPR: {best_metrics['fpr']:.4f} | FNR: {best_metrics['fnr']:.4f}")
        print(f"\nSaved analysis to: {outputs_dir / 'image_threshold_analysis.json'}")
        print(f"Saved CSV to:      {outputs_dir / 'image_threshold_analysis.csv'}")
        
        sys.exit(0)

    if args.smoke_test:
        print("\n" + "=" * 50)
        print("RUNNING SMOKE TEST")
        print("=" * 50)
        
        model.train()
        train_batches = 0
        train_loss = 0.0
        tensor_shape = None
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 17. Verify batch shape and labels
            assert inputs.shape[1:] == (3, 224, 224), f"Unexpected shape {inputs.shape}"
            assert torch.all((labels == 0) | (labels == 1)), "Labels must be 0 or 1"
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            train_batches += 1
            
            if train_batches == 1:
                tensor_shape = tuple(inputs.shape)
                
            if train_batches >= 2:
                break
                
        train_loss /= max(1, train_batches)
        
        # Validation pass (1 batch)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss = loss.item()
                break
                
        print(f"Class Mapping: {train_dataset.class_to_idx}")
        print(f"Tensor Shape:  {tensor_shape}")
        print(f"Batch Size:    {args.batch_size}")
        print(f"Device:        {device}")
        print(f"Train Loss:    {train_loss:.4f} (2 batches)")
        print(f"Val Loss:      {val_loss:.4f} (1 batch)")
        
        print("\n[SUCCESS] Image training smoke test passed.")
        sys.exit(0)

    print("\nStarting Training...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        
        all_train_preds = []
        all_train_labels = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_roc_auc, _, _ = evaluate_model(
            model, val_loader, criterion, device, use_amp
        )
        
        scheduler.step(val_f1) # Step scheduler based on F1
        
        # 18. Track metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)
        history['val_roc_auc'].append(val_roc_auc)
        
        print(f"Epoch {epoch+1}/{args.epochs} "
              f"| Train Loss: {epoch_train_loss:.4f} F1: {epoch_train_f1:.4f} "
              f"| Val Loss: {val_loss:.4f} F1: {val_f1:.4f} AUC: {val_roc_auc:.4f}")
        
        # 19. Save the best model based on validation F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (Val F1: {best_val_f1:.4f})")
            
        # Save last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'history': history,
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate()
        }
        torch.save(checkpoint, checkpoint_path)
            
    # 21. Save training history to: outputs/image_training_history.json
    with open(outputs_dir / "image_training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    # 25. Generate: outputs/image_training_curves.png
    plot_training_curves(history, outputs_dir / "image_training_curves.png")
    
    # ---------------------------------------------------------
    # 6. Evaluate on Test Set
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("EVALUATING BEST MODEL ON HELD-OUT TEST SET")
    print("=" * 50)
    
    # 28. Make sure the test set is never used during training or model selection.
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # 22. After training, evaluate the best model on the held-out test set.
    test_loss, test_acc, test_prec, test_rec, test_f1, test_roc_auc, test_labels, test_preds = evaluate_model(
        model, test_loader, criterion, device, use_amp
    )
    
    # 23. Print metrics
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")
    print(f"Test ROC-AUC:   {test_roc_auc:.4f}")
    
    test_metrics = {
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1_score": test_f1,
        "roc_auc": test_roc_auc
    }
    
    # 26. Save a JSON file: outputs/image_test_metrics.json
    with open(outputs_dir / "image_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    # 24. Generate: outputs/image_confusion_matrix.png
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, ["Real", "Fake"], outputs_dir / "image_confusion_matrix.png")
    
    print(f"\nTraining complete. Model saved to: {best_model_path}")
    print(f"Outputs saved to: {outputs_dir}")

# 29. Add argparse options
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeShield Image Forensics Training")
    parser.add_argument("--data-dir", type=str, default="data/processed/image", help="Path to the dataset directory")
    parser.add_argument("--output-dir", type=str, default="models/image", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16 for CPU friendliness)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--unfreeze", type=int, default=0, help="Number of final EfficientNet blocks to unfreeze (0 to freeze all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast smoke test (2 train batches, 1 val batch) without saving")
    parser.add_argument("--analyze-threshold", action="store_true", help="Run validation threshold analysis on the saved best model and exit")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    
    args = parser.parse_args()
    train(args)
