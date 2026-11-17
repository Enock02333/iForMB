import pandas as pd
from ifor_mb.backtracking.contextual import ContextEvent, rank_context
from ifor_mb.synthetic.features import add_temporal_and_numeric_features
from ifor_mb.synthetic.schema import resolve_schema


def test_synthetic_feature_pipeline():
    frame=pd.DataFrame({"timestamp":["2026-01-01T01:00:00Z"],"service":["api"],"request_id":["r"],"user":["u"],"message":["request failed"],"severity":["ERROR"],"client":["web"],"duration":[12]})
    schema=resolve_schema(frame.columns)
    transformed=add_temporal_and_numeric_features(frame,schema)
    assert {"hour_sin","hour_cos","duration_numeric","severity_ordinal"}.issubset(transformed.columns)


def test_context_ranker_prefers_operational_match():
    query=ContextEvent(3,"n1","api","E2","ERROR","timeout request",30)
    history=[ContextEvent(1,"n2","db","E1","INFO","healthy",10),ContextEvent(2,"n1","api","E2","ERROR","request timeout",20)]
    assert rank_context(query,history)[0].index==2
