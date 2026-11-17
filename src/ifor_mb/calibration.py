from __future__ import annotations
import numpy as np

def calibrate_isolation_scores(train_normal_scores, scores, epsilon=1e-9):
    """Return raw score, empirical percentile, and robust z-score."""
    reference = np.asarray(train_normal_scores, dtype=float)
    values = np.asarray(scores, dtype=float)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("train_normal_scores must be a non-empty 1D array")
    ordered = np.sort(reference)
    percentile = np.searchsorted(ordered, values, side="right") / ordered.size
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    robust_z = (values - median) / (1.4826 * mad + epsilon)
    return np.column_stack([values, percentile, robust_z])
