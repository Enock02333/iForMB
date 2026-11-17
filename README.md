# iForMB

Reference implementation for the iForMB manuscript on Isolation Forest evidence fusion, MLP anomaly detection, and contextual backtracking.

## Protocol

- Seeds: 13, 29, 47, 71, 101
- HDFS: stratified 60/20/20 and leave-one-failure-type-out
- BGL: non-overlapping 20-event windows and chronological 60/20/20
- Training-only preprocessing and calibration
- Validation-only model and threshold selection
- Post-detection backtracking with no relevance-label access during ranking

## Install

```bash
python -m venv .venv
pip install -e .[dev]
pytest
```

## Data

Raw datasets are not redistributed. See `data/README.md` and place files under `data/external/`.

```bash
ifor-mb verify-data --data-root data/external
```

## Traceability

See `docs/MANUSCRIPT_MAPPING.md` and the compact reference outputs in `results/reference/`.
