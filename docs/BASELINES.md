
# Matched external baselines

## DeepLog

The public architecture uses a training-only event vocabulary, normal-only sequence filtering, history length 10, a 32-dimensional embedding, and a 64-unit LSTM. Validation scores determine the operating threshold; test labels never select the threshold.

## LogRobust

The public architecture uses 300-dimensional semantic vectors, a bidirectional LSTM with 64 hidden units per direction, and attention pooling. The paper evaluation fitted FastText only on training-visible text and selected checkpoints and thresholds using validation data.

## Execution boundary

Full baseline re-training is an optional GPU workflow installed with `pip install -e .[baselines]`. `src/ifor_mb/baselines/runner.py` writes the complete contract before data access and refuses an unapproved full run. Compact verified summaries are committed under `results/reference/external-baselines-summary.csv`.
