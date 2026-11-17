# Outputs

Every experiment writes machine-readable artifacts under a user-selected output directory.

## Synthetic development pipeline

`synthetic-development-summary.json` records the split, seed, threshold, evidence dimensions, predicted-anomaly count, and the explicit non-ground-truth warning.

## HDFS and BGL

The experiment modules write per-seed JSON/CSV results containing configuration, threshold, anomaly-class metrics, and population counts. Compact paper-run summaries are provided in `results/reference/`.

## Backtracking

Outputs include Recall@1/3/5/10/20, MRR, mean first relevant rank, recoverable-query counts, context reduction, and latency.
