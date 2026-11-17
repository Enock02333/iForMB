# iForMB architecture

The repository implements the complete research pipeline:

1. **Data loading**: structured synthetic logs, HDFS block traces, or BGL event streams.
2. **Preprocessing**: time-cyclic, numerical, categorical, message, block, or window representations fitted from training data.
3. **Isolation evidence**: raw IF score, empirical normal percentile, and robust MAD-based z-score.
4. **Early feature-level fusion**: concatenate the calibrated evidence with the dataset-specific representation.
5. **MLP detection**: two hidden layers (128 and 64), Adam, L2 regularisation, and early stopping.
6. **Validation threshold**: maximise anomaly F1 subject to validation FPR <= 0.001.
7. **Anomaly decision**: apply the fixed threshold once to test data.
8. **Contextual backtracking**: for anomalous events, rank preceding records by node, component, type, severity, message overlap, and recency.

## Dataset roles

- The original synthetic dataset provides an executable end-to-end development demonstration. Its IF-derived labels are explicitly marked as a development oracle, not independent fault ground truth.
- HDFS provides conventional and unseen-failure-type labelled detection.
- BGL provides chronological labelled detection and contextual retrieval evaluation.
