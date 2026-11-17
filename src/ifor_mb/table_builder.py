
from __future__ import annotations

import csv
from pathlib import Path


def copy_reference_tables(reference_dir: Path, output_dir: Path):
    """Create generated table inputs from committed machine-readable summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for source in sorted(reference_dir.glob("*.csv")):
        rows = list(csv.reader(source.open("r", encoding="utf-8-sig", newline="")))
        if not rows:
            raise ValueError(f"Empty reference result: {source}")
        destination = output_dir / source.name
        with destination.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        generated.append(destination)
    if not generated:
        raise ValueError(f"No reference CSV files found in {reference_dir}")
    return generated


def verify_seed_contract(results, required=(13, 29, 47, 71, 101)):
    observed = sorted({int(row["seed"]) for row in results})
    if observed != list(required):
        raise ValueError(f"Expected seeds {list(required)}, observed {observed}")
    return True
