
import argparse
from pathlib import Path
from ifor_mb.table_builder import copy_reference_tables

p=argparse.ArgumentParser(description="Regenerate manuscript summary inputs from reference CSV files")
p.add_argument("--reference",type=Path,default=Path("results/reference"))
p.add_argument("--output",type=Path,default=Path("results/generated"))
a=p.parse_args()
for path in copy_reference_tables(a.reference,a.output):
    print(path)
