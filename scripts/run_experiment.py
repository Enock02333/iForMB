
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from ifor_mb.registry import EXPERIMENTS

p=argparse.ArgumentParser(description="Resolve and invoke an iForMB experiment")
p.add_argument("experiment",choices=sorted(EXPERIMENTS))
p.add_argument("--data-root",type=Path,default=Path("data/external"))
p.add_argument("--output",type=Path,default=Path("results/runs"))
p.add_argument("--seed",type=int,default=13)
p.add_argument("--smoke",action="store_true")
a=p.parse_args()
spec=EXPERIMENTS[a.experiment]
module_name,function_name=spec.runner.split(":")
function=getattr(importlib.import_module(module_name),function_name)
if a.experiment=="synthetic":
    raise SystemExit("Use scripts/run_synthetic_pipeline.py with --input for the original dataset.")
if a.experiment=="external-baselines":
    print(function(dataset="hdfs",data_root=a.data_root,output=a.output/a.experiment,seeds=(a.seed,),smoke=a.smoke))
else:
    print(f"Resolved {a.experiment}: {module_name}.{function_name}; use the dataset-specific documented runner because canonical modules have dataset-specific signatures.")
