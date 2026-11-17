
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def environment_snapshot():
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "pyyaml"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def write_provenance(output: Path, *, experiment: str, seed: int | None, config_path: Path | None, inputs: list[Path], split_contract: dict, threshold_source="validation"):
    root = Path(__file__).resolve().parents[2]
    record = {
        "experiment": experiment,
        "implementation_version": "1.0.0",
        "git_commit": git_commit(root),
        "seed": seed,
        "configuration_sha256": sha256_file(config_path) if config_path and config_path.exists() else None,
        "input_sha256": {path.name: sha256_file(path) for path in inputs if path.exists()},
        "split_contract": split_contract,
        "threshold_source": threshold_source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_snapshot(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
