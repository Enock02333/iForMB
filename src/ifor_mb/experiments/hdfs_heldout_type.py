from __future__ import annotations
import csv,json,re
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ifor_mb.evaluation.metrics import binary_metrics
SEEDS=[13,29,47,71,101]; EVENTS=[f"E{i}" for i in range(1,30)]; ER=re.compile(r"E\d+"); NR=re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def events(v): return ER.findall(str(v))
def nums(v): return [float(x) for x in NR.findall(str(v))]
def entropy(seq):
 if not seq:return 0.0
 c=np.array(list(Counter(seq).values()),float);p=c/c.sum();return float(-(p*np.log2(p)).sum())
def maxrun(seq):
 if not seq:return 0
 best=cur=1
 for a,b in zip(seq,seq[1:]):cur=cur+1 if a==b else 1;best=max(best,cur)
 return best
def stats(seq):
 a=np.asarray(seq,float)
 if not len(a):return [0,0,0,0,0,0]
 return [len(a),float(a.mean()),float(np.median(a)),float(a.std()),float(a.max()),float(np.quantile(a,.95))]
def engineer(matrix_path,traces_path,labels_path):
 m=pd.read_csv(matrix_path,usecols=["BlockId","Type",*EVENTS]);t=pd.read_csv(traces_path,usecols=["BlockId","Features","TimeInterval","Latency"]);l=pd.read_csv(labels_path)
 d=m.merge(t,on="BlockId",validate="one_to_one").merge(l,on="BlockId",validate="one_to_one")
 ev=d.Features.map(events);ints=d.TimeInterval.map(nums);base=np.log1p(d[EVENTS].to_numpy(float));extra=[]
 for e,iv,lat in zip(ev,ints,d.Latency.astype(float)):
  bi=list(zip(e,e[1:]));tri=list(zip(e,e[1:],e[2:]));adj=sum(a==b for a,b in bi);first=int(e[0][1:]) if e else 0;last=int(e[-1][1:]) if e else 0;s=stats(iv);duration=sum(iv)
  extra.append([len(e),len(set(e)),adj,maxrun(e),entropy(e),entropy(bi),entropy(tri),first,last,*s,float(lat),float(lat)/(len(e) or 1),len(e)/(duration+1.0)])
 y=(d.Label=="Anomaly").astype(int).to_numpy();typ=d.Type.to_numpy()
 return d,np.c_[base,np.asarray(extra,float)],y,typ
def best_threshold(y,s,max_fpr=.001):
 p,r,t=precision_recall_curve(y,s);f=2*p*r/np.maximum(p+r,1e-12);valid=[]
 for i,th in enumerate(t):
  pred=s>=th;fp=((pred==1)&(y==0)).sum();tn=((pred==0)&(y==0)).sum()
  if fp/max(fp+tn,1)<=max_fpr:valid.append(i)
 cand=np.asarray(valid if valid else range(len(t)));i=int(cand[np.nanargmax(f[cand])]);return float(t[i]),float(f[i])
def cal_if(Xtr,ytr,Xv,Xt,seed):
 iso=IsolationForest(n_estimators=300,max_samples=10000,contamination="auto",random_state=seed,n_jobs=-1).fit(Xtr[ytr==0])
 a=-iso.decision_function(Xtr);b=-iso.decision_function(Xv);c=-iso.decision_function(Xt);normal=a[ytr==0];med=np.median(normal);mad=np.median(np.abs(normal-med))+1e-9;sn=np.sort(normal)
 def tf(s):return np.c_[s,np.searchsorted(sn,s,side="right")/len(sn),(s-med)/(1.4826*mad)]
 return tf(a),tf(b),tf(c)
def balanced(y,idx,seed,ratio=3):
 rng=np.random.default_rng(seed);pos=idx[y[idx]==1];neg=idx[y[idx]==0];return rng.permutation(np.r_[pos,rng.choice(neg,min(len(neg),ratio*len(pos)),replace=False)])
def model(seed):return MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=1e-4,learning_rate_init=5e-4,early_stopping=True,validation_fraction=.1,n_iter_no_change=15,max_iter=300,random_state=seed)
def wilson(k,n,z=1.959963984540054):
 if n==0:return [None,None]
 p=k/n;den=1+z*z/n;ctr=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [float(max(0,ctr-half)),float(min(1,ctr+half))]
def run(matrix_path:Path,traces_path:Path,labels_path:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);d,X,y,typ=engineer(matrix_path,traces_path,labels_path);eligible=[float(x) for x,c in pd.Series(typ[y==1]).value_counts().items() if pd.notna(x) and c>=100];rows=[]
 for held in sorted(eligible):
  held_idx=np.where((y==1)&(typ==held))[0];available=np.where(~((y==1)&(typ==held)))[0]
  for seed in SEEDS:
   tr,va=train_test_split(available,test_size=.25,random_state=seed,stratify=y[available]);rng=np.random.default_rng(seed);normal_pool=va[y[va]==0];normal_test=rng.choice(normal_pool,min(len(normal_pool),max(10000,3*len(held_idx))),replace=False);va=np.setdiff1d(va,normal_test,assume_unique=False)
   sc=StandardScaler();A=sc.fit_transform(X[tr]);V=sc.transform(X[va]);test_idx=np.r_[held_idx,normal_test];T=sc.transform(X[test_idx]);IFa,IFv,IFt=cal_if(A,y[tr],V,T,seed);b=balanced(y,tr,seed);pos={g:i for i,g in enumerate(tr)};bi=np.array([pos[g] for g in b]);m=model(seed);m.fit(np.c_[A[bi],IFa[bi]],y[b]);sv=m.predict_proba(np.c_[V,IFv])[:,1];st=m.predict_proba(np.c_[T,IFt])[:,1];th,vf=best_threshold(y[va],sv);pred=st>=th;met=binary_metrics(y[test_idx],pred,st);tp=int(((pred==1)&(y[test_idx]==1)).sum());support=int((y[test_idx]==1).sum());met["heldout_recall_ci95_wilson"]=wilson(tp,support)
   rows.append({"held_out_type":held,"seed":seed,"support_anomaly":support,"support_normal":int((y[test_idx]==0).sum()),"threshold":th,"validation_f1":vf,"iterations":int(m.n_iter_),"metrics":met});(out/"partial-results.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
 summary=[]
 for held in sorted(eligible):
  q=[r for r in rows if r["held_out_type"]==held]
  for key in ["precision_anomaly","recall_anomaly","f1_anomaly","pr_auc","roc_auc","false_positive_rate","false_negative_rate"]:
   a=np.array([r["metrics"][key] for r in q],float);summary.append({"held_out_type":held,"metric":key,"mean":float(a.mean()),"std_sample":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max())})
 report={"experiment":"E4D leave-one-failure-type-out HDFS evaluation","architecture":"Fixed E4C canonical early fusion MLP [128,64], alpha=1e-4, lr=5e-4","protocol":"All anomalies of one failure type excluded from train/validation and used only in test; threshold selected on validation at FPR <= 0.001; seeded normal test cohort.","seeds":SEEDS,"eligible_types":sorted(eligible),"runs":rows,"summary":summary}
 (out/"hdfs-e4d-results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 with (out/"hdfs-e4d-summary.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["held_out_type","metric","mean","std_sample","min","max"]);w.writeheader();w.writerows(summary)
 return report
