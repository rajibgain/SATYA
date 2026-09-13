import numpy as np
from src.audio.evaluate import calculate_metrics

def test_eer():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.6, 0.9]
    y_pred = [0, 0, 1, 1]
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    eer = metrics['eer']
    print("Test passed! Calculated EER:", eer)
    
    # EER should be 0.0 with perfect separation
    assert eer is not None
    assert abs(eer) < 1e-6

if __name__ == "__main__":
    test_eer()
