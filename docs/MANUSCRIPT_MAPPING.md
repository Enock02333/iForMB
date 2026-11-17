# Manuscript-to-code mapping

| Manuscript element | Implementation |
|---|---|
| IF evidence calibration | `src/ifor_mb/calibration.py` |
| Validation threshold | `src/ifor_mb/evaluation/thresholds.py` |
| Conventional HDFS | `src/ifor_mb/experiments/hdfs_conventional.py` |
| HDFS unseen type | `src/ifor_mb/experiments/hdfs_matched_unseen.py` |
| Chronological BGL | `src/ifor_mb/experiments/bgl_chronological.py` |
| Backtracking | `src/ifor_mb/backtracking/bgl_backtracking.py` |
| DeepLog | `src/ifor_mb/baselines/deeplog.py` |
| LogRobust | `src/ifor_mb/baselines/logrobust.py` |
| Original dataset | `src/ifor_mb/synthetic/demonstration.py` |

| Fused detector | `src/ifor_mb/models.py` |
| Full synthetic pipeline | `src/ifor_mb/synthetic/pipeline.py` |
| Reusable contextual ranker | `src/ifor_mb/backtracking/contextual.py` |
