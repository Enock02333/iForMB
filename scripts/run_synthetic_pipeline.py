import argparse
from pathlib import Path
from ifor_mb.synthetic.pipeline import run_synthetic_pipeline

p=argparse.ArgumentParser(description="Run the original-dataset iForMB development pipeline")
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seed", type=int, default=13)
a=p.parse_args()
print(run_synthetic_pipeline(a.input, a.output, a.seed))
