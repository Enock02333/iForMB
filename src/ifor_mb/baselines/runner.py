
from __future__ import annotations

import json
from pathlib import Path

from .deeplog import make_next_event_examples


def run_external_baselines(*, dataset: str, data_root: Path, output: Path, seeds=(13, 29, 47, 71, 101), smoke=False):
    """Public matched-baseline orchestration contract.

    Full baseline training requires the ``baselines`` extra. The function records the
    architecture/training contract and rejects an unsupported dataset before expensive work.
    The canonical publication summaries remain under results/reference.
    """
    if dataset not in {"hdfs", "bgl"}:
        raise ValueError("dataset must be 'hdfs' or 'bgl'")
    try:
        import torch  # noqa: F401
        import gensim  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install baseline dependencies with: pip install -e .[baselines]") from error
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "dataset": dataset,
        "seeds": list(seeds),
        "deeplog": {"history": 10, "embedding": 32, "hidden": 64, "training": "normal-only next-event"},
        "logrobust": {"semantic_dim": 300, "hidden_per_direction": 64, "representation": "training-only FastText", "attention": True},
        "threshold_source": "validation",
        "max_fpr": 0.001,
        "smoke": bool(smoke),
        "data_root": str(data_root),
    }
    (output / "external-baseline-contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if not smoke:
        raise RuntimeError("Full external-baseline execution is intentionally explicit: use the paper configuration and dataset adapters documented in docs/BASELINES.md. The contract file was written; no test data were opened.")
    return contract
