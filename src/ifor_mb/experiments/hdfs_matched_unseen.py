from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .e4d_heldout_type import engineer,best_threshold,balanced,model,SEEDS
from ifor_mb.evaluation.metrics import binary_metrics
METHODS=("isolation_forest","enriched_mlp","early_if_mlp")
BUDGETS=(0.001,0.005,0.01)
def if_features(Xtr,ytr,Xv,Xt,seed):
 iso=IsolationForest(n_estimators=300,max_samples=10000,contamination="auto",random_state=seed,n_jobs=-1).fit(Xtr[ytr==0])
 tr=-iso.decision_function(Xtr);va=-iso.decision_function(Xv);te=-iso.decision_function(Xt)
 normal=tr[ytr==0]; med=np.median(normal); mad=np.median(np.abs(normal-med))+1e-9; sn=np.sort(normal)
 def tf(s): return np.c_[s,np.searchsorted(sn,s,side="right")/len(sn),(s-med)/(1.4826*mad)]
 return tr,va,te,tf(tr),tf(va),tf(te)
def alert_budget_metrics(y,s,budget):
 n=max(1,int(np.ceil(len(s)*budget))); idx=np.argsort(s)[-n:]; p=np.zeros(len(s),dtype=bool);p[idx]=True
 m=binary_metrics(y,p,s);m["alert_budget"]=budget;m["alerts"]=n;return m
def paired_summary(rows):
 out=[]
 for method in METHODS:
  q=[r for r in rows if r["method"]==method]
  for key in ("precision_anomaly","recall_anomaly","f1_anomaly","pr_auc","roc_auc","false_positive_rate","false_negative_rate"):
   a=np.array([r["metrics"][key] for r in q],float);out.append({"method":method,"metric":key,"mean":float(a.mean()),"std_sample":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"n":len(a)})
  recalls=np.array([r["metrics"]["recall_anomaly"] for r in q],float);out.append({"method":method,"metric":"catastrophic_recall_lt_0_10","mean":float((recalls<.10).mean()),"std_sample":None,"min":None,"max":None,"n":len(recalls)})
 return out
def run(matrix_path:Path,traces_path:Path,labels_path:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);d,X,y,typ=engineer(matrix_path,traces_path,labels_path)
 eligible=[float(x) for x,c in pd.Series(typ[y==1]).value_counts().items() if pd.notna(x) and c>=100];rows=[]
 for held in sorted(eligible):
  held_idx=np.where((y==1)&(typ==held))[0];available=np.where(~((y==1)&(typ==held)))[0]
  for seed in SEEDS:
   tr,va=train_test_split(available,test_size=.25,random_state=seed,stratify=y[available]);rng=np.random.default_rng(seed);normal_pool=va[y[va]==0];normal_test=rng.choice(normal_pool,min(len(normal_pool),max(10000,3*len(held_idx))),replace=False);va=np.setdiff1d(va,normal_test,assume_unique=False);test_idx=np.r_[held_idx,normal_test]
   sc=StandardScaler();A=sc.fit_transform(X[tr]);V=sc.transform(X[va]);T=sc.transform(X[test_idx]);rawA,rawV,rawT,IFA,IFV,IFT=if_features(A,y[tr],V,T,seed)
   b=balanced(y,tr,seed);pos={g:i for i,g in enumerate(tr)};bi=np.array([pos[g] for g in b]);scores={"isolation_forest":(rawV,rawT)}
   plain=model(seed);plain.fit(A[bi],y[b]);scores["enriched_mlp"]=(plain.predict_proba(V)[:,1],plain.predict_proba(T)[:,1])
   early=model(seed);early.fit(np.c_[A[bi],IFA[bi]],y[b]);scores["early_if_mlp"]=(early.predict_proba(np.c_[V,IFV])[:,1],early.predict_proba(np.c_[T,IFT])[:,1])
   for method,(sv,st) in scores.items():
    th,vf=best_threshold(y[va],sv);pred=st>=th;met=binary_metrics(y[test_idx],pred,st)
    budgets={str(b):alert_budget_metrics(y[test_idx],st,b) for b in BUDGETS}
    rows.append({"held_out_type":held,"seed":seed,"method":method,"support_anomaly":int((y[test_idx]==1).sum()),"support_normal":int((y[test_idx]==0).sum()),"threshold":th,"validation_f1":vf,"metrics":met,"fixed_alert_budgets":budgets})
   (out/"partial-results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
 summary=paired_summary(rows);report={"experiment":"E4E matched held-out-failure-type baseline comparison","protocol":"Same E4D cohorts, seeds, preprocessing, normal-only Isolation Forest fit, validation threshold rule FPR <= 0.001, and fixed E4C MLP architecture for all supervised methods.","methods":list(METHODS),"fixed_alert_budgets":list(BUDGETS),"eligible_types":sorted(eligible),"seeds":SEEDS,"runs":rows,"aggregate_summary":summary}
 (out/"hdfs-e4e-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 with (out/"hdfs-e4e-summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["method","metric","mean","std_sample","min","max","n"]);w.writeheader();w.writerows(summary)
 return report
