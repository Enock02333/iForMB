from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="ifor-mb")
    parser.add_argument("command", choices=["verify-data", "synthetic"])
    parser.add_argument("--data-root", type=Path, default=Path("data/external"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()

    if args.command == "verify-data":
        required = [
            args.data_root / "hdfs" / "Event_occurrence_matrix.csv",
            args.data_root / "hdfs" / "Event_traces.csv",
            args.data_root / "hdfs" / "anomaly_label.csv",
            args.data_root / "bgl" / "BGL.log",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SystemExit("Missing external data:\n" + "\n".join(missing))
        print("Data contract satisfied.")
        return

    if args.input is None:
        raise SystemExit("--input is required")
    from .synthetic.demonstration import describe_framework_data

    print(describe_framework_data(args.input))


if __name__ == "__main__":
    main()
