"""Offline: service rank of the root cause in logs / trace_latency / trace_errors rankings, v1 vs v2 vs v3 tiers."""
import json, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import RobustScaler
from RCAEval.agent.toolkit import CaseToolkit, robust_score, sustained_step, is_quiet
T=sys.argv[1]
A=[json.loads(l) for l in open(f"agent_tasks/{T}.answers.jsonl")]
def svc_rank(names, s):
    seen=list(dict.fromkeys(n.split("_")[0].lower() for n in names)); return seen.index(s)+1 if s in seen else 99
res={}
for a in A:
    tk=CaseToolkit(f"agent_tasks/{T}/cases/{a['case_id']}")
    for mod,src in (("logs","logts"),("trace_latency","tracets_lat"),("trace_errors","tracets_err")):
        if not tk._has(src): continue
        d=tk.series(src); d=d[(d.time>=tk.window["start"])&(d.time<tk.window["end"])]
        n,an=d[d.time<tk.inject_time],d[d.time>=tk.inject_time]
        if n.empty or an.empty: continue
        rows={"v1":[], "v2":[], "v3":[]}
        for c in d.columns:
            if c=="time" or d[c].nunique()<=1: continue
            v1=float(np.max(RobustScaler().fit(n[[c]].to_numpy()).transform(an[[c]].to_numpy())))
            v2=robust_score(n[c],an[c],d[c])
            st=sustained_step(n[c].to_numpy(),an[c].to_numpy())
            k3=(2,st["rise"],v2) if st else ((0,0,v2) if is_quiet(n[c].to_numpy()) else (1,0,v2))
            rows["v1"].append((c,(v1,))); rows["v2"].append((c,(v2,))); rows["v3"].append((c,k3))
        for v,r in rows.items():
            names=[c for c,_ in sorted(r,key=lambda x:x[1],reverse=True)]
            sr=svc_rank(names,a["service"]); key=(mod,v)
            res.setdefault(key,[0,0,0]); res[key][0]+=sr==1; res[key][1]+=sr<=3; res[key][2]+=1
for (mod,v),(s1,s3,n) in sorted(res.items()): print(f"{T} {mod:14s} {v}: svc@1={s1:3d} svc@3={s3:3d} / {n}")
