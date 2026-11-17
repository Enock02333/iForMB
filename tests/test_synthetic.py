import pandas as pd
from ifor_mb.synthetic.demonstration import describe_framework_data

def test_synthetic_role(tmp_path):
    path=tmp_path/"synthetic.csv"
    pd.DataFrame({"timestamp":["t"],"service":["s"],"request_id":["r"],"user":["u"],"message":["m"],"severity":["INFO"],"client":["c"],"duration":[1]}).to_csv(path,index=False)
    assert describe_framework_data(path)["independent_binary_ground_truth"] is False
