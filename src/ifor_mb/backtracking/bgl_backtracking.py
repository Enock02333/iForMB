from __future__ import annotations
import csv,json,re,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
PAT=re.compile(r'^(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s?(.*)$')
LOOKBACK=100; KS=(1,3,5,10,20)
def tokens(s): return set(re.findall(r'(?u)\b\w+\b',s.lower()))
def parse(path):
 out=[]
 with path.open('r',encoding='utf-8',errors='replace') as f:
  for ln,line in enumerate(f,1):
   m=PAT.match(line.rstrip('\n'))
   if not m:raise ValueError(f'parse failure line {ln}')
   label,epoch,date,node,fullts,node2,typ,comp,level,msg=m.groups();out.append({'label':label,'epoch':int(epoch),'node':node,'type':typ,'component':comp,'level':level,'message':msg,'tokens':tokens(msg),'line':ln})
 return out
def onset_indices(ev,start):
 q=[];prev='-'
 for i in range(start,len(ev)):
  lab=ev[i]['label']
  if lab!='-' and lab!=prev:q.append(i)
  prev=lab
 return q
def rankers(query,cands):
 def jac(a,b):
  u=a|b;return len(a&b)/len(u) if u else 0.0
 scored={'recency':[], 'node_only':[], 'component_only':[], 'contextual':[]}
 for j,x in cands:
  d=query['line']-x['line'];rec=1/max(d,1);same_node=float(query['node']==x['node']);same_comp=float(query['component']==x['component']);same_type=float(query['type']==x['type']);same_level=float(query['level']==x['level']);sim=jac(query['tokens'],x['tokens'])
  scored['recency'].append((rec,j));scored['node_only'].append((3*same_node+rec,j));scored['component_only'].append((2*same_comp+rec,j));scored['contextual'].append((3*same_node+2*same_comp+same_type+.5*same_level+2*sim+rec,j))
 return {k:[j for _,j in sorted(v,key=lambda z:(z[0],z[1]),reverse=True)] for k,v in scored.items()}
def summarize(rows,method,eligible_only=False):
 q=[r for r in rows if (r['recoverable'] or not eligible_only)];N=len(q);ranks=[r['ranks'][method] for r in q if r['ranks'][method] is not None]
 d={'queries':N,'recoverable_queries':sum(r['recoverable'] for r in q),'mrr':sum((1/x if x else 0) for x in [r['ranks'][method] for r in q])/N if N else None,'mean_first_rank':float(np.mean(ranks)) if ranks else None,'median_first_rank':float(np.median(ranks)) if ranks else None}
 for k in KS:d[f'recall_at_{k}']=sum((r['ranks'][method] is not None and r['ranks'][method]<=k) for r in q)/N if N else None
 d['context_reduction_at_5']=1-5/LOOKBACK;return d
def run(path:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True);ev=parse(path);start=int(.8*len(ev));onsets=onset_indices(ev,start);rows=[];lat=defaultdict(list)
 for i in onsets:
  q=ev[i];c=[(j,ev[j]) for j in range(max(start,i-LOOKBACK),i)];relevant={j for j,x in c if x['label']==q['label']};t0=time.perf_counter_ns();ranked=rankers(q,c);elapsed=(time.perf_counter_ns()-t0)/1e6
  ranks={m:(min((p+1 for p,j in enumerate(order) if j in relevant),default=None)) for m,order in ranked.items()}
  rows.append({'query_line':q['line'],'label':q['label'],'recoverable':bool(relevant),'relevant_prior_count':len(relevant),'candidate_count':len(c),'ranks':ranks,'ranking_latency_ms_total':elapsed})
 methods=['recency','node_only','component_only','contextual'];overall={m:{'all_onsets':summarize(rows,m,False),'recoverable_only':summarize(rows,m,True)} for m in methods};by_label={}
 for lab,count in Counter(r['label'] for r in rows).most_common():
  z=[r for r in rows if r['label']==lab];by_label[lab]={'support':len(z),'recoverable':sum(r['recoverable'] for r in z),'methods':{m:summarize(z,m,False) for m in methods}}
 report={'experiment':'E6 BGL backtracking matched evaluation','source':str(path),'test_start_event':start,'lookback_events':LOOKBACK,'query_definition':'onset of a consecutive run of the same non-dash BGL label','relevance':'prior event with the same externally supplied BGL label within the frozen 100-event history','ranking_label_guard':'BGL labels are used only after ranking to score relevance; no ranker uses labels','methods':{'recency':'most recent first','node_only':'same node then recency','component_only':'same component then recency','contextual':'same node, component, event type, severity level, message-token Jaccard similarity, and recency'},'queries':len(rows),'recoverable_queries':sum(r['recoverable'] for r in rows),'overall':overall,'by_label':by_label,'latency_ms':{'mean_total_per_query':float(np.mean([r['ranking_latency_ms_total'] for r in rows])),'p95_total_per_query':float(np.quantile([r['ranking_latency_ms_total'] for r in rows],.95))},'claim_boundary':'Evaluates prioritisation of prior same-category operational evidence at episode onset; it does not establish causal root-cause identification.','rows':rows}
 (out/'bgl-e6-backtracking-results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 with (out/'bgl-e6-backtracking-summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['method','cohort','metric','value'])
  for m in methods:
   for cohort,z in overall[m].items():
    for k,v in z.items():w.writerow([m,cohort,k,v])
 return report
