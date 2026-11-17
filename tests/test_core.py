import numpy as np
from ifor_mb.calibration import calibrate_isolation_scores
from ifor_mb.evaluation.thresholds import select_threshold

def test_calibration_and_threshold():
    evidence=calibrate_isolation_scores([0.1,0.2,0.3,0.4],[0.15,0.35])
    assert evidence.shape==(2,3) and evidence[0,1] < evidence[1,1]
    selected=select_threshold(np.array([0,0,0,0,1,1]),np.array([0.01,0.02,0.03,0.04,0.8,0.9]))
    assert selected["validation_fpr"] <= 0.001
