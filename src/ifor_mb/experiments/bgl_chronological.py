from __future__ import annotations
import csv,json,re
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from ifor_mb.evaluation.metrics import binary_metrics
SEEDS=[13,29,47,71,101]; WINDOW=20
PAT=re.compile(r'^(\S+)\s+(\d+)\s+(\d{4}\.\d{2}\.\d{2})\s+(\S+)\s+(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s?(.*)$')
def entropy(v):
 c=np.asarray(list(Counter(v).values()),float)
 if not len(c):return 0.0
 p=c/c.sum();return float(-(p*np.log2(p)).sum())
def build(path:Path,index_path:Path):
 hv=HashingVectorizer(n_features=256,alternate_sign=False,norm=None,token_pattern=r'(?u)\b\w+\b'); docs=[]; extra=[]; y=[]; index=[]; buf=[]
 def flush():
  if len(buf)!=WINDOW:return
  labels,epochs,nodes,types,comps,levels,msgs,lines=zip(*buf);dur=max(0,epochs[-1]-epochs[0]);anom=sum(labels)
  docs.append(' '.join(msgs));extra.append([len(set(nodes)),len(set(types)),len(set(comps)),len(set(levels)),entropy(types),entropy(comps),entropy(levels),dur,anom/WINDOW])
  y.append(int(anom>0));index.append({'window_id':len(index),'start_line':lines[0],'end_line':lines[-1],'start_epoch':epochs[0],'end_epoch':epochs[-1],'anomaly_events':anom,'label':int(anom>0),'nodes':list(nodes),'types':list(types),'components':list(comps),'levels':list(levels),'messages':list(msgs)})
 with path.open('r',encoding='utf-8',errors='replace') as f:
  for line_no,line in enumerate(f,1):
   m=PAT.match(line.rstrip('\n'))
   if not m:raise ValueError(f'BGL parse failure at line {line_no}')
   lab,ep,date,node,fullts,node2,typ,comp,lvl,msg=m.groups();buf.append((lab!='-',int(ep),node,typ,comp,lvl,msg,line_no))
   if len(buf)==WINDOW:flush();buf=[]
 Xtext=hv.transform(docs).toarray();X=np.c_[np.log1p(Xtext),np.asarray(extra,float)]
 index_path.write_text(json.dumps({'window_size':WINDOW,'stride':WINDOW,'windows':index},indent=2),encoding='utf-8')
 return X,np.asarray(y,int)
def best_threshold(y,s,max_fpr=.001):
 p,r,t=precision_recall_curve(y,s);f=2*p*r/np.maximum(p+r,1e-12);valid=[]
 for i,th in enumerate(t):
  pred=s>=th;fp=((pred==1)&(y==0)).sum();tn=((pred==0)&(y==0)).sum()
  if fp/max(fp+tn,1)<=max_fpr:valid.append(i)
 cand=np.asarray(valid if valid else range(len(t)));i=int(cand[np.nanargmax(f[cand])]);return float(t[i]),float(f[i])
def cal_if(A,y,V,T,seed):
 iso=IsolationForest(n_estimators=300,max_samples=min(10000,int((y==0).sum())),contamination='auto',random_state=seed,n_jobs=-1).fit(A[y==0]);a=-iso.decision_function(A);v=-iso.decision_function(V);t=-iso.decision_function(T);n=a[y==0];med=np.median(n);mad=np.median(np.abs(n-med))+1e-9;sn=np.sort(n)
 def z(s):return np.c_[s,np.searchsorted(sn,s,side='right')/len(sn),(s-med)/(1.4826*mad)]
 return z(a),z(v),z(t),a,v,t
def balanced(y,seed,ratio=3):
 rng=np.random.default_rng(seed);p=np.where(y==1)[0];n=np.where(y==0)[0];return rng.permutation(np.r_[p,rng.choice(n,min(len(n),ratio*len(p)),replace=False)])
def model(seed):return MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',solver='adam',alpha=1e-4,learning_rate_init=5e-4,early_stopping=True,validation_fraction=.1,n_iter_no_change=15,max_iter=300,random_state=seed)
def aggregate(rows,name,key):
 a=np.asarray([r['methods'][name]['metrics'][key] for r in rows],float);return {'mean':float(a.mean()),'std_sample':float(a.std(ddof=1)),'min':float(a.min()),'max':float(a.max())}
def run(path:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);X,y=build(path,out/'bgl-window-index.json');n=len(y);a=int(.6*n);b=int(.8*n);tr=np.arange(a);va=np.arange(a,b);te=np.arange(b,n);rows=[]
 for seed in SEEDS:
  sc=StandardScaler();A=sc.fit_transform(X[tr]);V=sc.transform(X[va]);T=sc.transform(X[te]);IFa,IFv,IFt,rawa,rawv,rawt=cal_if(A,y[tr],V,T,seed);bi=balanced(y[tr],seed);methods={}
  ith,ivf=best_threshold(y[va],rawv);methods['isolation_forest']={'threshold':ith,'validation_f1':ivf,'metrics':binary_metrics(y[te],rawt>=ith,rawt)}
  m=model(seed);m.fit(A[bi],y[tr][bi]);sv=m.predict_proba(V)[:,1];st=m.predict_proba(T)[:,1];th,vf=best_threshold(y[va],sv);methods['enriched_mlp']={'threshold':th,'validation_f1':vf,'iterations':int(m.n_iter_),'metrics':binary_metrics(y[te],st>=th,st)}
  h=model(seed);h.fit(np.c_[A[bi],IFa[bi]],y[tr][bi]);hv=h.predict_proba(np.c_[V,IFv])[:,1];ht=h.predict_proba(np.c_[T,IFt])[:,1];hth,hvf=best_threshold(y[va],hv);methods['early_if_mlp']={'threshold':hth,'validation_f1':hvf,'iterations':int(h.n_iter_),'metrics':binary_metrics(y[te],ht>=hth,ht)}
  rows.append({'seed':seed,'split':{'train':len(tr),'validation':len(va),'test':len(te),'train_anomaly':int(y[tr].sum()),'validation_anomaly':int(y[va].sum()),'test_anomaly':int(y[te].sum())},'methods':methods});(out/'partial-results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 names=['isolation_forest','enriched_mlp','early_if_mlp'];keys=['accuracy','precision_anomaly','recall_anomaly','f1_anomaly','pr_auc','roc_auc','false_positive_rate','false_negative_rate'];summary={n:{k:aggregate(rows,n,k) for k in keys} for n in names}
 report={'experiment':'E5 BGL matched chronological labelled validation','dataset':str(path),'window_contract':{'size':20,'stride':20,'label':'any first-token non-dash event','split':'contiguous chronological 60/20/20'},'feature_contract':'256-dimensional stateless message hashing plus window node/type/component/level diversity and entropy, duration, and anomaly-event fraction excluded from predictors','architecture':'Fixed MLP [128,64], alpha=1e-4, lr=5e-4; early fusion adds calibrated IF raw/percentile/robust-z features','selection':'Threshold selected on validation only subject to FPR <= 0.001; test used once','seeds':SEEDS,'runs':rows,'summary':summary}
 (out/'bgl-e5-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 with (out/'bgl-e5-summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['method','metric','mean','std_sample','min','max'])
  for name,z in summary.items():
   for key,v in z.items():w.writerow([name,key,v['mean'],v['std_sample'],v['min'],v['max']])
 return report
