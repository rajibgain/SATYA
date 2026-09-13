import torch
import json
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculates binary classification metrics.
    Positive class (1) = AI-Generated.
    """
    # Convert to numpy for easier handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        else:
            roc_auc = None  # Single class edge case
    except ImportError:
        roc_auc = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
    }

def save_evaluation(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
