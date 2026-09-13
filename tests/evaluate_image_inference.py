import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.image.detector import ImageForensicsDetector
from src.image.preprocess import preprocess_image

def main():
    print("Initializing ImageForensicsDetector...")
    try:
        detector = ImageForensicsDetector()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        sys.exit(1)
        
    test_real_dir = Path("data/processed/image/test/real")
    test_fake_dir = Path("data/processed/image/test/fake")
    
    if not test_real_dir.exists() or not test_fake_dir.exists():
        print("Error: Test directories do not exist.")
        sys.exit(1)
    
    true_labels = []
    pred_labels = []
    fake_confidences = []
    misclassified = []
    
    # Label mapping requirement: real = 0, fake = 1
    
    def process_directory(directory, true_label):
        images = list(directory.glob("*.*"))
        for img_path in images:
            # Skip non-image files if any
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
                
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = preprocess_image(img)
                result = detector.predict(tensor)
                
                # Result contains label ("Fake" or "Real") and confidence
                pred_label_str = result["label"]
                conf = result["confidence"]
                
                # Convert prediction string to binary label (real = 0, fake = 1)
                pred_label = 1 if pred_label_str.lower() == "fake" else 0
                
                # Determine probability of Fake (class 1)
                prob_fake = conf if pred_label == 1 else (1.0 - conf)
                
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                fake_confidences.append(prob_fake)
                
                if pred_label != true_label:
                    misclassified.append({
                        "file": img_path.name,
                        "true": "Real" if true_label == 0 else "Fake",
                        "pred": pred_label_str,
                        "conf": conf
                    })
            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")
                
    print("Processing real images (True Label = 0)...")
    process_directory(test_real_dir, 0)
    
    print("Processing fake images (True Label = 1)...")
    process_directory(test_fake_dir, 1)
    
    if not true_labels:
        print("No images were processed.")
        sys.exit(1)
        
    total_real = sum(1 for label in true_labels if label == 0)
    total_fake = sum(1 for label in true_labels if label == 1)
    total = len(true_labels)
    
    print("\n--- Summary ---")
    print(f"Total real images: {total_real}")
    print(f"Total fake images: {total_fake}")
    print(f"Total images evaluated: {total}")
    
    # Calculate metrics
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(true_labels, fake_confidences)
    except ValueError:
        roc_auc = 0.0
        
    print("\n--- Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    real_pred_fake = cm[0][1]
    fake_pred_real = cm[1][0]
    
    print("\n--- Errors ---")
    print(f"Real images predicted fake (False Positives): {real_pred_fake}")
    print(f"Fake images predicted real (False Negatives): {fake_pred_real}")
    
    print("\nFirst 10 misclassified files:")
    for idx, item in enumerate(misclassified[:10]):
        print(f"  {idx+1}. {item['file']} - True: {item['true']}, Pred: {item['pred']} ({item['conf']:.4f})")
        
    # Generate confusion matrix plot using matplotlib only
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Independent Inference Confusion Matrix')
    plt.colorbar()
    
    classes = ['Real (0)', 'Fake (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90, va='center')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "inference_confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    
    print(f"\nConfusion matrix saved to: {out_path}")
    print("\n[PASS] Independent inference evaluation completed.")

if __name__ == "__main__":
    main()
