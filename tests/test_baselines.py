from ifor_mb.baselines.deeplog import make_next_event_examples

def test_deeplog_examples():
    x,y=make_next_event_examples([[1,2,3,4]],history=2)
    assert x.tolist()==[[1,2],[2,3]] and y.tolist()==[3,4]
