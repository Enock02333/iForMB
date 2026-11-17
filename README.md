# iForMB

**Integrated Isolation Forest evidence, multilayer-perceptron anomaly detection, and contextual backtracking for distributed-system logs.**

![iForMB architecture](docs/assets/ifor-mb-architecture.jpg)

## Research contribution

A standalone Isolation Forest supplies an unsupervised unusualness score, but that score is not a task-specific anomaly boundary. iForMB transforms the isolation response into three evidence features:

- raw Isolation Forest score;
- percentile within the normal training-score distribution;
- robust z-score computed from the median and MAD of normal training scores.

The evidence is concatenated with the dataset-specific log representation **before** MLP classification. The final threshold is selected on validation data by maximising anomaly F1 subject to `FPR <= 0.001`. For anomalous decisions, contextual backtracking ranks prior operational evidence without using relevance labels during ranking.

## Complete pipeline

```text
raw logs
  -> schema validation and parsing
  -> training-fitted feature construction
  -> normal-only Isolation Forest
  -> calibrated IF evidence [raw, percentile, robust-z]
  -> early feature-level fusion
  -> MLP anomaly probability
  -> validation-selected low-FPR threshold
  -> anomaly decision
       -> if anomalous: contextual backtracking
       -> ranked prior operational evidence
```

## Dataset roles

| Dataset | Role | Implemented workflow |
|---|---|---|
| Synthetic Log Data of Distributed System | Original framework-development dataset | Structured schema, temporal/categorical/message features, IF development oracle, calibrated evidence, MLP fusion, thresholding, output summary |
| HDFS_v1 | Independently labelled detector evaluation | Conventional stratified testing and leave-one-failure-type-out stress testing |
| BGL | Independently labelled temporal evaluation | Non-overlapping 20-event chronological windows and contextual-backtracking evaluation |

The synthetic workflow is executable, but its IF-derived development targets are not represented as independent ground truth. HDFS and BGL provide the labelled performance evidence.

## Repository structure

```text
src/ifor_mb/
  calibration.py                 # raw, percentile, robust-z IF evidence
  models.py                      # IFMLPDetector
  synthetic/                     # original-dataset schema, features, full pipeline
  experiments/                   # HDFS and BGL evaluation regimes
  baselines/                     # matched DeepLog and LogRobust architectures
  backtracking/                  # BGL experiment and reusable context ranker
  evaluation/                    # metrics and constrained thresholding
scripts/
  run_synthetic_pipeline.py
configs/
results/reference/
docs/
tests/
```

## Installation

```bash
python -m venv .venv
# Windows
.venv/Scripts/python -m pip install -e .[dev]
# Linux/macOS
.venv/bin/python -m pip install -e .[dev]
pytest
```

Install `.[baselines]` to instantiate the PyTorch DeepLog and LogRobust models.

## Original synthetic-dataset pipeline

Download the Kaggle dataset identified in `data/README.md`, then run:

```bash
python scripts/run_synthetic_pipeline.py \
  --input data/external/synthetic/synthetic_logs.csv \
  --output results/runs/synthetic \
  --seed 13
```

The command produces `synthetic-development-summary.json` and explicitly labels the IF-derived targets as development targets rather than external ground truth.

## HDFS and BGL protocols

- Seeds: `13, 29, 47, 71, 101`
- HDFS: stratified `60/20/20` and leave-one-failure-type-out
- BGL: non-overlapping 20-event windows and chronological `60/20/20`
- Training-only preprocessing, vocabularies, semantic adapters, Isolation Forest, and calibration
- Validation-only checkpoint and threshold selection
- One-time test evaluation
- Backtracking relevance categories excluded from the ranking function

See [Reproducibility](docs/REPRODUCIBILITY.md), [Architecture](docs/ARCHITECTURE.md), [Outputs](docs/OUTPUTS.md), and [Manuscript mapping](docs/MANUSCRIPT_MAPPING.md).

## Reference results

`results/reference/` contains compact summaries for conventional HDFS, unseen-type HDFS, chronological BGL, backtracking, and matched external baselines. Raw logs, the large BGL window index, and checkpoints are intentionally excluded.

## Scope

The implementation supports anomaly detection and prioritisation of prior same-category operational evidence. Contextual backtracking is post-detection and does not alter classifier metrics. The repository does not claim causal root-cause identification or production deployment validation.
