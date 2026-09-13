import argparse
import random
import json
import csv
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Import inference functions to reuse logic
from src.image.inference import load_model, analyze_image

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation Sanity Test")
    parser.add_argument("--data-dir", type=str, default="data/processed/image_ntire/test", help="Path to test dataset directory")
    parser.add_argument("--model-path", type=str, default="models/image_ntire_unfrozen/best_image_model.pth", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.53, help="Decision threshold")
    parser.add_argument("--num-real", type=int, default=100, help="Number of real images to evaluate")
    parser.add_argument("--num-fake", type=int, default=100, help="Number of fake images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic image selection")
    parser.add_argument("--output-json", type=str, default="outputs/image_batch_sanity_results.json", help="Path to save detailed JSON results")
    parser.add_argument("--output-csv", type=str, default="outputs/image_batch_sanity_results.csv", help="Path to save detailed CSV results")
    
    args = parser.parse_args()
    
    # 5. Use deterministic selection (seed 42)
    random.seed(args.seed)
    
    data_path = Path(args.data_dir)
    real_dir = data_path / "real"
    fake_dir = data_path / "fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        print(f"[ERROR] Data directories not found. Looked for:\n  {real_dir}\n  {fake_dir}")
        sys.exit(1)
        
    # Get sorted lists of images to ensure consistent random sampling across runs
    real_images = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
    fake_images = sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
    
    num_real = min(args.num_real, len(real_images))
    num_fake = min(args.num_fake, len(fake_images))
    
    selected_real = random.sample(real_images, num_real)
    selected_fake = random.sample(fake_images, num_fake)
    
    print(f"Selected {num_real} real images and {num_fake} fake images for evaluation.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
        
    results = []
    y_true = []
    y_pred = []
    y_probs = []
    
    # Process Real Images (Class 0)
    print("Evaluating Real images...")
    for i, img_path in enumerate(selected_real):
        if (i+1) % 25 == 0 or (i+1) == len(selected_real):
            print(f"  Processed {i+1}/{len(selected_real)} real images...")
            
        res = analyze_image(str(img_path), model, device, args.threshold)
        pred_label = 1 if res['verdict'] == "LIKELY FAKE" else 0
        correct = (pred_label == 0)
        
        results.append({
            "filename": img_path.name,
            "true_label": "REAL",
            "predicted_label": res['verdict'].replace("LIKELY ", ""),
            "fake_probability": res['fake_probability'],
            "real_probability": res['real_probability'],
            "correct": correct
        })
        y_true.append(0)
        y_pred.append(pred_label)
        y_probs.append(res['fake_probability'])
        
    # Process Fake Images (Class 1)
    print("Evaluating Fake images...")
    for i, img_path in enumerate(selected_fake):
        if (i+1) % 25 == 0 or (i+1) == len(selected_fake):
            print(f"  Processed {i+1}/{len(selected_fake)} fake images...")
            
        res = analyze_image(str(img_path), model, device, args.threshold)
        pred_label = 1 if res['verdict'] == "LIKELY FAKE" else 0
        correct = (pred_label == 1)
        
        results.append({
            "filename": img_path.name,
            "true_label": "FAKE",
            "predicted_label": res['verdict'].replace("LIKELY ", ""),
            "fake_probability": res['fake_probability'],
            "real_probability": res['real_probability'],
            "correct": correct
        })
        y_true.append(1)
        y_pred.append(pred_label)
        y_probs.append(res['fake_probability'])
        
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    real_correct = int(tn)
    real_incorrect = int(fp)
    fake_correct = int(tp)
    fake_incorrect = int(fn)
    
    real_accuracy = real_correct / (real_correct + real_incorrect) if (real_correct + real_incorrect) > 0 else 0
    fake_accuracy = fake_correct / (fake_correct + fake_incorrect) if (fake_correct + fake_incorrect) > 0 else 0
    
    # Prepare summary output
    summary = {
        "metrics": {
            "overall_accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "real_accuracy": float(real_accuracy),
            "fake_accuracy": float(fake_accuracy)
        },
        "breakdown": {
            "real_correctly_classified": real_correct,
            "real_incorrectly_classified_as_fake": real_incorrect,
            "fake_correctly_classified": fake_correct,
            "fake_incorrectly_classified_as_real": fake_incorrect
        },
        "settings": {
            "model": args.model_path,
            "threshold": args.threshold,
            "num_real": num_real,
            "num_fake": num_fake,
            "seed": args.seed
        }
    }
    
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    # Save CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["filename", "true_label", "predicted_label", "fake_probability", "real_probability", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\n" + "=" * 50)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall:           {rec:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print("-" * 50)
    print("CLASS BREAKDOWN:")
    print(f"Real Images (Total: {real_correct + real_incorrect})")
    print(f"  Accuracy:                 {real_accuracy:.4f}")
    print(f"  Correctly Classified:     {real_correct}")
    print(f"  Incorrectly labeled FAKE: {real_incorrect}")
    print(f"Fake Images (Total: {fake_correct + fake_incorrect})")
    print(f"  Accuracy:                 {fake_accuracy:.4f}")
    print(f"  Correctly Classified:     {fake_correct}")
    print(f"  Incorrectly labeled REAL: {fake_incorrect}")
    print("-" * 50)
    print(f"Detailed JSON saved to: {json_path}")
    print(f"Detailed CSV saved to:  {csv_path}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
