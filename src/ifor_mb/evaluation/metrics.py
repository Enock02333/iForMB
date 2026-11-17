from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score, roc_auc_score

def binary_metrics(y_true, y_pred, scores=None) -> dict:
    y_true=np.asarray(y_true).astype(int); y_pred=np.asarray(y_pred).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=[1],zero_division=0)
    out={"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"support_normal":int((y_true==0).sum()),
         "support_anomaly":int((y_true==1).sum()),"precision_anomaly":float(p[0]),"recall_anomaly":float(r[0]),
         "f1_anomaly":float(f[0]),"false_positive_rate":float(fp/(fp+tn)) if fp+tn else 0.0,
         "false_negative_rate":float(fn/(fn+tp)) if fn+tp else 0.0,"accuracy":float((tp+tn)/len(y_true))}
    if scores is not None and len(np.unique(y_true))==2:
        out["pr_auc"]=float(average_precision_score(y_true,scores)); out["roc_auc"]=float(roc_auc_score(y_true,scores))
    return out
