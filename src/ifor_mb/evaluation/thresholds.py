from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_curve

def select_threshold(y_validation, scores, max_fpr=0.001):
    """Maximise validation F1 subject to an FPR constraint."""
    y = np.asarray(y_validation, dtype=int)
    values = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, values)
    candidates = []
    for i, threshold in enumerate(thresholds):
        prediction = values >= threshold
        fp = np.sum((prediction == 1) & (y == 0))
        tn = np.sum((prediction == 0) & (y == 0))
        fpr = fp / max(fp + tn, 1)
        f1 = 2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1e-12)
        if fpr <= max_fpr:
            candidates.append((float(f1), -float(fpr), float(threshold)))
    if not candidates:
        raise ValueError("No validation threshold satisfies max_fpr")
    f1, neg_fpr, threshold = max(candidates)
    return {"threshold": threshold, "validation_f1": f1, "validation_fpr": -neg_fpr}
