from __future__ import annotations
import csv,json,math,re,time
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ifor_mb.evaluation.metrics import binary_metrics
SEEDS=[13,29,47,71,101]; EVENTS=[f"E{i}" for i in range(1,30)]
ER=re.compile(r"E\d+"); NR=re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def events(v): return ER.findall(str(v))
def nums(v): return [float(x) for x in NR.findall(str(v))]
def entropy(seq):
 if not seq:return 0.0
 c=np.array(list(Counter(seq).values()),float); p=c/c.sum(); return float(-(p*np.log2(p)).sum())
def maxrun(seq):
 if not seq:return 0
 best=cur=1
 for a,b in zip(seq,seq[1:]): cur=cur+1 if a==b else 1; best=max(best,cur)
 return best
def stats(seq):
 a=np.asarray(seq,float)
 if not len(a):return [0,0,0,0,0,0]
 return [len(a),float(a.mean()),float(np.median(a)),float(a.std()),float(a.max()),float(np.quantile(a,.95))]
def engineer(matrix_path,traces_path,labels_path):
 m=pd.read_csv(matrix_path,usecols=["BlockId",*EVENTS]); t=pd.read_csv(traces_path,usecols=["BlockId","Features","TimeInterval","Latency"]); l=pd.read_csv(labels_path)
 d=m.merge(t,on="BlockId",validate="one_to_one").merge(l,on="BlockId",validate="one_to_one")
 ev=d.Features.map(events); ints=d.TimeInterval.map(nums)
 base=np.log1p(d[EVENTS].to_numpy(float)); extra=[]
 for e,iv,lat in zip(ev,ints,d.Latency.astype(float)):
  bi=list(zip(e,e[1:])); tri=list(zip(e,e[1:],e[2:])); adj=sum(a==b for a,b in bi)
  first=int(e[0][1:]) if e else 0; last=int(e[-1][1:]) if e else 0
  s=stats(iv); duration=sum(iv); rate=len(e)/(duration+1.0)
  extra.append([len(e),len(set(e)),adj,maxrun(e),entropy(e),entropy(bi),entropy(tri),first,last,*s,float(lat),float(lat)/(len(e) or 1),rate])
 return d,np.c_[base,np.asarray(extra,float)],(d.Label=="Anomaly").astype(int).to_numpy()
def best_threshold(y,s,max_fpr=None):
 p,r,t=precision_recall_curve(y,s); f=2*p*r/np.maximum(p+r,1e-12); cand=np.arange(len(t))
 if max_fpr is not None:
  valid=[]
  for i,th in enumerate(t):
   pred=s>=th; fp=((pred==1)&(y==0)).sum(); tn=((pred==0)&(y==0)).sum(); fpr=fp/max(fp+tn,1)
   if fpr<=max_fpr:valid.append(i)
  if valid:cand=np.asarray(valid)
 i=int(cand[np.nanargmax(f[cand])]);return float(t[i]),float(f[i])
def cal_if(Xtr,ytr,Xv,Xt,seed):
 iso=IsolationForest(n_estimators=300,max_samples=10000,contamination="auto",random_state=seed,n_jobs=-1).fit(Xtr[ytr==0])
 a=-iso.decision_function(Xtr);b=-iso.decision_function(Xv);c=-iso.decision_function(Xt);normal=a[ytr==0];med=np.median(normal);mad=np.median(np.abs(normal-med))+1e-9;sortedn=np.sort(normal)
 def transform(s):
  pct=np.searchsorted(sortedn,s,side="right")/len(sortedn);z=(s-med)/(1.4826*mad);return np.c_[s,pct,z]
 return transform(a),transform(b),transform(c),a,b,c
def curves(y,s):
 p,r,t=precision_recall_curve(y,s);f,tpr,th=roc_curve(y,s);return {"pr":{"precision":p.tolist(),"recall":r.tolist(),"thresholds":t.tolist()},"roc":{"fpr":f.tolist(),"tpr":tpr.tolist(),"thresholds":th.tolist()}}
def balanced(y,idx,seed,ratio=3):
 rng=np.random.default_rng(seed);pos=idx[y[idx]==1];neg=idx[y[idx]==0];return rng.permutation(np.r_[pos,rng.choice(neg,min(len(neg),ratio*len(pos)),replace=False)])
def model(cfg,seed):
 return MLPClassifier(hidden_layer_sizes=tuple(cfg["layers"]),activation="relu",solver="adam",alpha=cfg["alpha"],learning_rate_init=cfg["lr"],early_stopping=True,validation_fraction=.1,n_iter_no_change=15,max_iter=300,random_state=seed)
def aggregate(rows,name,key):
 a=np.array([r[name]["metrics"][key] for r in rows],float);return {"mean":float(a.mean()),"std_sample":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max())}
def run(matrix_path:Path,traces_path:Path,labels_path:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);d,X,y=engineer(matrix_path,traces_path,labels_path);idx=np.arange(len(y))
 # Validation-only architecture selection on seed 13. Test partition is never scored here.
 tr,tmp=train_test_split(idx,test_size=.4,random_state=13,stratify=y);va,_=train_test_split(tmp,test_size=.5,random_state=13,stratify=y[tmp]);sc=StandardScaler();A=sc.fit_transform(X[tr]);V=sc.transform(X[va]);IFa,IFv,_,rawa,rawv,_=cal_if(A,y[tr],V,V,13);b=balanced(y,tr,13);pos={g:i for i,g in enumerate(tr)};bi=np.array([pos[g] for g in b])
 configs=[{"name":"canonical","layers":[128,64],"alpha":1e-4,"lr":5e-4},{"name":"deep","layers":[256,128,64],"alpha":1e-4,"lr":5e-4},{"name":"regularized","layers":[128,64,32],"alpha":1e-3,"lr":5e-4}]
 tuning=[]
 for cfg in configs:
  for fusion in ["enriched_mlp","early_if_mlp"]:
   xa=A[bi] if fusion=="enriched_mlp" else np.c_[A[bi],IFa[bi]];xv=V if fusion=="enriched_mlp" else np.c_[V,IFv]
   m=model(cfg,13);m.fit(xa,y[b]);sv=m.predict_proba(xv)[:,1];th,vf=best_threshold(y[va],sv,max_fpr=.001)
   tuning.append({"config":cfg,"fusion":fusion,"validation_threshold":th,"validation_f1_at_fpr_le_0.001":vf,"validation_metrics":binary_metrics(y[va],sv>=th,sv),"iterations":int(m.n_iter_)})
 tuning.sort(key=lambda x:(x["validation_f1_at_fpr_le_0.001"],x["validation_metrics"]["pr_auc"]),reverse=True);winner=tuning[0]
 (out/"validation-tuning.json").write_text(json.dumps({"selection_rule":"Highest validation F1 subject to FPR <= 0.001; PR-AUC tie-breaker.","candidates":tuning,"winner":winner},indent=2),encoding="utf-8")
 rows=[]
 for seed in SEEDS:
  tr,tmp=train_test_split(idx,test_size=.4,random_state=seed,stratify=y);va,te=train_test_split(tmp,test_size=.5,random_state=seed,stratify=y[tmp]);sc=StandardScaler();A=sc.fit_transform(X[tr]);V=sc.transform(X[va]);T=sc.transform(X[te]);IFa,IFv,IFt,rawa,rawv,rawt=cal_if(A,y[tr],V,T,seed);b=balanced(y,tr,seed);pos={g:i for i,g in enumerate(tr)};bi=np.array([pos[g] for g in b]);cfg=winner["config"]
  methods={}
  # enriched MLP baseline
  m=model(cfg,seed);m.fit(A[bi],y[b]);sv=m.predict_proba(V)[:,1];st=m.predict_proba(T)[:,1];th,vf=best_threshold(y[va],sv,max_fpr=.001);methods["enriched_mlp"]={"threshold":th,"validation_f1":vf,"metrics":binary_metrics(y[te],st>=th,st),"curves":curves(y[te],st),"iterations":int(m.n_iter_)}
  # early fusion
  h=model(cfg,seed);h.fit(np.c_[A[bi],IFa[bi]],y[b]);hv=h.predict_proba(np.c_[V,IFv])[:,1];ht=h.predict_proba(np.c_[T,IFt])[:,1];hth,hvf=best_threshold(y[va],hv,max_fpr=.001);methods["early_if_mlp"]={"threshold":hth,"validation_f1":hvf,"metrics":binary_metrics(y[te],ht>=hth,ht),"curves":curves(y[te],ht),"iterations":int(h.n_iter_)}
  # gated late fusion, alpha selected on validation only
  ifprob=1/(1+np.exp(-IFv[:,2]));ifprobt=1/(1+np.exp(-IFt[:,2]));best=None
  for alpha in [.6,.7,.8,.9]:
   gv=alpha*sv+(1-alpha)*ifprob;gth,gf=best_threshold(y[va],gv,max_fpr=.001)
   candidate=(gf,alpha,gth,gv)
   if best is None or candidate[0]>best[0]:best=candidate
  gf,alpha,gth,gv=best;gt=alpha*st+(1-alpha)*ifprobt;methods["gated_if_mlp"]={"alpha":alpha,"threshold":gth,"validation_f1":gf,"metrics":binary_metrics(y[te],gt>=gth,gt),"curves":curves(y[te],gt)}
  ith,ivf=best_threshold(y[va],rawv,max_fpr=.001);methods["isolation_forest"]={"threshold":ith,"validation_f1":ivf,"metrics":binary_metrics(y[te],rawt>=ith,rawt),"curves":curves(y[te],rawt)}
  row={"seed":seed,"split":{"train":len(tr),"validation":len(va),"test":len(te),"test_anomaly":int(y[te].sum())},"selected_config":cfg,**methods};rows.append(row);(out/"partial-results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
 names=["isolation_forest","enriched_mlp","early_if_mlp","gated_if_mlp"];keys=["accuracy","precision_anomaly","recall_anomaly","f1_anomaly","pr_auc","roc_auc","false_positive_rate","false_negative_rate"]
 summary={n:{k:aggregate(rows,n,k) for k in keys} for n in names}
 report={"experiment":"E4C tuned proposed framework on independently labelled HDFS blocks","feature_contract":"log1p E1-E29 plus sequence, interval, latency, and event-rate statistics; calibrated IF raw/percentile/robust-z features","leakage_exclusions":["BlockId","embedded labels","failure Type","external label"],"selection":"Architecture selected on validation seed 13 only; threshold and gated alpha selected independently within each validation split; test used once.","winner":winner,"seeds":SEEDS,"runs":rows,"summary":summary}
 (out/"hdfs-e4c-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 with (out/"hdfs-e4c-summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(["method","metric","mean","std_sample","min","max"])
  for n,z in summary.items():
   for k,v in z.items():w.writerow([n,k,v["mean"],v["std_sample"],v["min"],v["max"]])
 return report
