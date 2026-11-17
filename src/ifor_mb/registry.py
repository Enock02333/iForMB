
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config: str
    runner: str
    description: str


EXPERIMENTS = {
    "synthetic": ExperimentSpec("synthetic", "configs/synthetic.yaml", "ifor_mb.synthetic.pipeline:run_synthetic_pipeline", "Original structured-log development pipeline"),
    "hdfs-conventional": ExperimentSpec("hdfs-conventional", "configs/hdfs_conventional.yaml", "ifor_mb.experiments.hdfs_conventional:run", "Five-seed stratified HDFS evaluation"),
    "hdfs-unseen-type": ExperimentSpec("hdfs-unseen-type", "configs/hdfs_unseen_type.yaml", "ifor_mb.experiments.hdfs_matched_unseen:run", "Leave-one-failure-type-out HDFS evaluation"),
    "bgl-chronological": ExperimentSpec("bgl-chronological", "configs/bgl_chronological.yaml", "ifor_mb.experiments.bgl_chronological:run", "Chronological BGL evaluation"),
    "bgl-backtracking": ExperimentSpec("bgl-backtracking", "configs/bgl_backtracking.yaml", "ifor_mb.backtracking.bgl_backtracking:run", "BGL contextual retrieval evaluation"),
    "external-baselines": ExperimentSpec("external-baselines", "configs/external_baselines.yaml", "ifor_mb.baselines.runner:run_external_baselines", "Matched DeepLog and LogRobust evaluation"),
}
