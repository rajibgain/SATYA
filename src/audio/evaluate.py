import numpy as np

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculates classification metrics for audio detection.
    """
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

    eer = None
    eer_threshold = 0.5
    from sklearn.metrics import roc_curve
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = interp1d(fpr, thresholds)(eer)
        eer = float(eer)
        eer_threshold = float(eer_threshold)

    real_recall = tn / (tn + fp + 1e-9)
    fake_recall = tp / (tp + fn + 1e-9)
    balanced_accuracy = (real_recall + fake_recall) / 2

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "balanced_accuracy": float(balanced_accuracy),
        "real_recall": float(real_recall),
        "fake_recall": float(fake_recall),
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
    }
