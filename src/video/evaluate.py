"""Evaluation utilities for the SATYA video forensics skeleton."""

from typing import Any, Dict, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def evaluate_predictions(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute standard metrics for binary classification."""
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    if labels.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "threshold": threshold,
            "predictions": [],
            "true_labels": [],
            "probabilities": [],
        }

    predictions = (probs >= threshold).astype(int)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": None,
        "threshold": threshold,
        "predictions": predictions.tolist(),
        "true_labels": labels.tolist(),
        "probabilities": probs.tolist(),
    }

    unique_labels = np.unique(labels)
    if unique_labels.size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics
