
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]
reference=ROOT/"results/reference"
required={
    "hdfs-conventional-summary.csv",
    "hdfs-unseen-type-summary.csv",
    "bgl-chronological-summary.csv",
    "bgl-backtracking-summary.csv",
    "external-baselines-summary.csv",
}
observed={path.name for path in reference.glob("*.csv")}
missing=sorted(required-observed)
if missing:
    raise SystemExit(f"Missing reference summaries: {missing}")
for path in reference.glob("*.csv"):
    rows=list(csv.reader(path.open("r",encoding="utf-8-sig",newline="")))
    if len(rows)<2:
        raise SystemExit(f"Reference summary has no data rows: {path}")
print("Reference result contracts passed.")
