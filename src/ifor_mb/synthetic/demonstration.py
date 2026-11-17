from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "service", "request_id", "user", "message", "severity", "client", "duration"}

def load_synthetic_log(path: Path):
    frame = pd.read_csv(path)
    normalized = {column.lower().strip().replace(" ", "_") for column in frame.columns}
    missing = sorted(REQUIRED_COLUMNS.difference(normalized))
    if missing:
        raise ValueError(f"Missing synthetic-dataset columns: {missing}")
    return frame

def describe_framework_data(path: Path):
    frame = load_synthetic_log(path)
    severity = next(column for column in frame.columns if column.lower().strip() == "severity")
    return {"records": int(len(frame)), "columns": list(frame.columns), "severity_counts": frame[severity].value_counts(dropna=False).to_dict(), "role": "framework development and workflow demonstration", "independent_binary_ground_truth": False}
