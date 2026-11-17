
from pathlib import Path
import pandas as pd
import yaml
from ifor_mb.registry import EXPERIMENTS
from ifor_mb.synthetic.schema import resolve_schema
from ifor_mb.table_builder import copy_reference_tables

ROOT=Path(__file__).resolve().parents[1]

def test_experiment_registry_is_complete():
    assert set(EXPERIMENTS)=={"synthetic","hdfs-conventional","hdfs-unseen-type","bgl-chronological","bgl-backtracking","external-baselines"}

def test_synthetic_fixture_contract():
    frame=pd.read_csv(ROOT/"tests/fixtures/synthetic-mini.csv")
    schema=resolve_schema(frame.columns)
    assert schema.message=="message" and len(frame)==6

def test_bgl_protocol_guards():
    c=yaml.safe_load((ROOT/"configs/bgl_chronological.yaml").read_text())
    assert c["window"]=={"size":20,"stride":20}
    assert c["split"]["chronological"] is True

def test_reference_table_regeneration(tmp_path):
    generated=copy_reference_tables(ROOT/"results/reference",tmp_path)
    assert len(generated)==5
    assert all(path.stat().st_size>0 for path in generated)
