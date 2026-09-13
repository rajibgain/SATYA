import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
from dataset import AIDetectionDataset, get_transforms
from model import AIDetector
import time

def train_model(data_dir="data/processed/ai_detection/train", epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please run download_dataset.py first.")
        return

    full_dataset = AIDetectionDataset(data_dir, transform=get_transforms(train=True))
    total_size = len(full_dataset)
    
    if total_size == 0:
        print("No images found in dataset. Please run download_dataset.py first.")
        return
        
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = get_transforms(train=False) # Important: Val transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Training on {train_size} images, validating on {val_size} images.")

    # 2. Model, Loss, Optimizer
    model = AIDetector(freeze_blocks=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 3. Training Loop
    best_val_auc = 0.0
    patience = 2
    patience_counter = 0

    os.makedirs("models/ai_detection", exist_ok=True)
    save_path = "models/ai_detection/best_ai_detector.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # 4. Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # Handle cases where batch is too small and only has one class
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0
            
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] Summary ({epoch_time:.1f}s):")
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()
