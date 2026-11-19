"""HMM/Viterbi route decoder and held-out student-test evaluation.

Prediction reads candidate features and model scores, plus .track/.nodes/.arcs.
The .route files are read only in the separate eval-label section, after
prediction, to produce held-out comparison labels. They never enter emissions
or route decoding.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets"/"processed_v1"
RESULTS=ROOT/"datasets"/"model_results"
STUDENT=ROOT/"student_test"
LABELS_OUT=DATA/"student_test_eval_labels.csv.gz"
PRED_OUT=RESULTS/"hmm_viterbi_student_test_predictions.csv.gz"
SUMMARY_OUT=RESULTS/"hmm_viterbi_summary.csv"
ROUTES_OUT=RESULTS/"hmm_viterbi_route_sequences.json"

# Decoder settings to report and tune after the initial comparison.
MAX_STATES=20
GPS_GAP_RESET_S=120.0
PATH_CUTOFF_M=8000.0
NO_PATH_LOG_PENALTY=-25.0
SEED=42


def read_table(path,names):
    return pd.read_csv(path,sep=r"\s+",header=None,names=names,dtype=float)


def project(lon,lat,lon0,lat0):
    # Local azimuthal-equidistant metric projection used for the route alignment.
    R=6371008.8
    lon=np.radians(np.asarray(lon,float)); lat=np.radians(np.asarray(lat,float))
    lon0r,lat0r=np.radians(lon0),np.radians(lat0); dl=lon-lon0r
    cc=np.sin(lat0r)*np.sin(lat)+np.cos(lat0r)*np.cos(lat)*np.cos(dl)
    c=np.arccos(np.clip(cc,-1,1)); sc=np.sin(c)
    k=np.ones_like(c); mask=np.abs(sc)>1e-12; k[mask]=c[mask]/sc[mask]
    x=R*k*np.cos(lat)*np.sin(dl)
    y=R*k*(np.cos(lat0r)*np.sin(lat)-np.sin(lat0r)*np.cos(lat)*np.cos(dl))
    return np.column_stack([x,y])


def align_reference_route(track,nodes,arcs,route_ids):
    """Return one route-ordered reference arc per alignable GPS observation."""
    n=len(track); target=np.full(n,-1,dtype=np.int64)
    valid_ids=[int(a) for a in route_ids if np.isfinite(a) and a==int(a) and 0<=int(a)<len(arcs)]
    valid_ids=[a for a in valid_ids if 0<=int(arcs[a,0])<len(nodes) and 0<=int(arcs[a,1])<len(nodes)]
    if not n or not valid_ids: return target
    lon0=float(track[:,0].mean()); lat0=float(track[:,1].mean())
    gps_xy=project(track[:,0],track[:,1],lon0,lat0)
    nodes_xy=project(nodes[:,0],nodes[:,1],lon0,lat0)
    uv=arcs[np.asarray(valid_ids,dtype=int)].astype(int)
    a=nodes_xy[uv[:,0]]; b=nodes_xy[uv[:,1]]; vec=b-a
    lengths=np.linalg.norm(vec,axis=1)
    starts=np.r_[0.0,np.cumsum(lengths[:-1])]
    states=[]
    for point in gps_xy:
        den=np.einsum("ij,ij->i",vec,vec)
        frac=np.divide(np.einsum("ij,ij->i",point-a,vec),den,out=np.zeros_like(den),where=den>0)
        frac=np.clip(frac,0,1); proj=a+frac[:,None]*vec
        dist=np.linalg.norm(point-proj,axis=1)
        order=np.argsort(dist,kind="stable")
        within=order[dist[order]<=250.0]
        order=within if len(within) else order[:min(3,len(order))]
        order=order[:16]
        states.append([(int(j),float(starts[j]+frac[j]*lengths[j]),float(dist[j])) for j in order])
    step=np.zeros(n)
    if n>1: step[1:]=np.linalg.norm(np.diff(gps_xy,axis=0),axis=1)
    # Run the monotonic route-progress dynamic program separately across gaps
    # where no reference route segment is geographically close.
    i=0
    while i<n:
        while i<n and not states[i]: i+=1
        if i>=n: break
        j=i
        while j+1<n and states[j+1]: j+=1
        back=[np.full(len(states[k]),-1,dtype=np.int16) for k in range(i,j+1)]
        prev_cost=None; prev_prog=None
        for k in range(i,j+1):
            row=states[k]; prog=np.array([x[1] for x in row]); emission=np.array([x[2]/20.0 for x in row])
            if prev_cost is None: cost=emission.copy()
            else:
                cost=np.full(len(row),np.inf)
                for q,p in enumerate(prog):
                    allowed=np.flatnonzero(prev_prog<=p+1e-6)
                    if not len(allowed): allowed=np.arange(len(prev_prog)); trans=np.full(len(allowed),20.0)
                    else: trans=np.minimum(np.abs((p-prev_prog[allowed])-max(float(step[k]),0.0))/60.0,25.0)
                    vals=prev_cost[allowed]+trans; z=int(np.argmin(vals))
                    cost[q]=emission[q]+vals[z]; back[k-i][q]=int(allowed[z])
            prev_cost=cost; prev_prog=prog
        q=int(np.argmin(prev_cost))
        for k in range(j,i-1,-1):
            row=states[k]; q=min(q,len(row)-1); route_pos=row[q][0]
            target[k]=valid_ids[route_pos]
            if k>i and back[k-i][q]>=0: q=int(back[k-i][q])
            else: q=0
        i=j+1
    return target


class RoadGraph:
    def __init__(self,nodes,arcs):
        self.nodes=nodes; self.arcs=arcs; self.n=len(nodes)
        lon0=float(np.mean(nodes[:,0])); lat0=float(np.mean(nodes[:,1]))
        xy=project(nodes[:,0],nodes[:,1],lon0,lat0)
        self.edge_len=np.zeros(len(arcs),dtype=float)
        best={}
        self.valid=np.zeros(len(arcs),dtype=bool)
        for aid,(uf,vf) in enumerate(arcs):
            u,v=int(uf),int(vf)
            if not(0<=u<self.n and 0<=v<self.n): continue
            length=float(np.linalg.norm(xy[v]-xy[u]))
            self.edge_len[aid]=length; self.valid[aid]=True
            if (u,v) not in best or length<best[(u,v)][0]: best[(u,v)]=(length,aid)
        self.best_arc={k:v[1] for k,v in best.items()}
        rows=[k[0] for k in best]; cols=[k[1] for k in best]; weights=[v[0] for v in best.values()]
        self.graph=csr_matrix((weights,(rows,cols)),shape=(self.n,self.n),dtype=np.float64)
        self._source=self._cached_source

    @lru_cache(maxsize=128)
    def _cached_source(self,src):
        return dijkstra(self.graph,directed=True,indices=int(src),return_predecessors=True,limit=PATH_CUTOFF_M)

    def distance(self,src,dst):
        if src==dst:return 0.0
        if not(0<=src<self.n and 0<=dst<self.n):return np.inf
        dist,_=self._source(int(src))
        return float(dist[dst])

    def path_arcs(self,src,dst):
        if src==dst:return []
        dist,pred=self._source(int(src))
        if not np.isfinite(dist[dst]):return None
        nodes=[int(dst)]; cur=int(dst)
        while cur!=int(src):
            cur=int(pred[cur])
            if cur<0 or len(nodes)>self.n:return None
            nodes.append(cur)
        nodes.reverse()
        return [self.best_arc.get((u,v)) for u,v in zip(nodes[:-1],nodes[1:])]


def transition_distance(graph,p,q):
    if p["arc_id"]==q["arc_id"]:
        if q["fraction"]+1e-6<p["fraction"]: return np.inf
        return max(0.0,(q["fraction"]-p["fraction"])*p["length"])
    middle=graph.distance(p["v"],q["u"])
    if not np.isfinite(middle):return np.inf
    return max(0,(1-p["fraction"])*p["length"])+middle+max(0,q["fraction"]*q["length"])


def emission_log_probs(states):
    score=np.asarray([max(float(s["score"]),1e-12) for s in states])
    p=score/score.sum()
    return np.log(np.clip(p,1e-12,1.0))


def decode_sequence(groups,graph):
    """Viterbi over the candidate arcs at each GPS observation."""
    chosen=[]; boundary=[0]
    for i in range(1,len(groups)):
        dt=float(groups[i][0]["dt"])
        if not np.isfinite(dt) or dt>GPS_GAP_RESET_S: boundary.append(i)
    boundary.append(len(groups))
    for lo,hi in zip(boundary[:-1],boundary[1:]):
        if lo>=hi:continue
        dp=emission_log_probs(groups[lo]); backs=[]
        for t in range(lo+1,hi):
            prev,cur=groups[t-1],groups[t]
            emit=emission_log_probs(cur); trans=np.empty((len(prev),len(cur)))
            step=max(float(cur[0]["gps_step"]),0.0); dt=max(float(cur[0]["dt"]),1.0)
            scale=max(30.0,0.35*step+8.0*dt)
            for a,p in enumerate(prev):
                for b,q in enumerate(cur):
                    road=transition_distance(graph,p,q)
                    trans[a,b]=NO_PATH_LOG_PENALTY if not np.isfinite(road) else -abs(road-step)/scale
            values=dp[:,None]+trans; ptr=np.argmax(values,axis=0)
            dp=values[ptr,np.arange(len(cur))]+emit; backs.append(ptr)
        state=int(np.argmax(dp)); path=[state]
        for ptr in reversed(backs): state=int(ptr[state]); path.append(state)
        path.reverse()
        for k,t in enumerate(range(lo,hi)):
            chosen.append((t,path[k]))
    chosen.sort()
    return [groups[t][s] for t,s in chosen]


def expand_route(states,graph):
    route=[]; broken=0
    if not states:return route,broken
    route=[states[0]["arc_id"]]
    for p,q in zip(states[:-1],states[1:]):
        if p["arc_id"]==q["arc_id"] and q["fraction"]>=p["fraction"]:
            continue
        bridge=graph.path_arcs(p["v"],q["u"])
        if bridge is None or any(a is None for a in bridge):
            broken+=1; route.append(q["arc_id"]); continue
        for aid in bridge+[q["arc_id"]]:
            if not route or route[-1]!=aid: route.append(aid)
    return route,broken


def main():
    score_path=RESULTS/"xgboost_tuned_student_test_candidate_scores.csv.gz"
    if not score_path.exists(): score_path=RESULTS/"xgboost_student_test_candidate_scores.csv.gz"
    features_path = DATA / "student_test_candidates.csv"
    scores=pd.read_csv(score_path,dtype={"trip_id":str})
    feat=pd.read_csv(features_path,dtype={"trip_id":str})
    data=feat.merge(scores[["trip_id","observation_idx","candidate_arc_id","candidate_score"]],
        on=["trip_id","observation_idx","candidate_arc_id"],how="inner",validate="one_to_one")
    if len(data)!=len(scores): print("Warning: candidate feature/score row counts differ.")
    prediction_rows=[]; summaries=[]; route_outputs=[]; eval_label_rows=[]
    for trip_id,frame in data.groupby("trip_id",sort=True):
        folder=STUDENT/trip_id; prefix=folder/trip_id
        track_df=read_table(Path(str(prefix)+".track"),["longitude","latitude","timestamp_s"])
        nodes_df=read_table(Path(str(prefix)+".nodes"),["longitude","latitude"])
        arcs_df=read_table(Path(str(prefix)+".arcs"),["from_node_id","to_node_id"])
        track=track_df[["longitude","latitude","timestamp_s"]].to_numpy(float)
        nodes=nodes_df[["longitude","latitude"]].to_numpy(float)
        arcs=arcs_df[["from_node_id","to_node_id"]].to_numpy(float)
        graph=RoadGraph(nodes,arcs)

        # Prediction-only stage: candidates, GPS, network, and model scores; no .route read.
        grouped=[]
        for obs,part in frame.groupby("observation_idx",sort=True):
            part=part.sort_values("candidate_score",ascending=False)
            states=[]
            for row in part.itertuples(index=False):
                aid=int(row.candidate_arc_id)
                if not(0<=aid<len(arcs)) or not graph.valid[aid]:continue
                u,v=map(int,arcs[aid])
                states.append({"trip_id":trip_id,"obs":int(obs),"arc_id":aid,"u":u,"v":v,
                    "fraction":float(np.clip(row.projection_fraction,0,1)),
                    "length":float(graph.edge_len[aid]),"score":float(row.candidate_score),
                    "rank":int(row.candidate_rank),"distance":float(row.candidate_distance_m),
                    "gps_step":float(row.gps_step_m),"dt":float(row.time_delta_s)})
            if states:grouped.append(states)
        decoded=decode_sequence(grouped,graph)
        lookup={(s["obs"],s["arc_id"]):s for group in grouped for s in group}
        by_obs={}
        for s in decoded:
            by_obs[s["obs"]]=s
            prediction_rows.append({"trip_id":trip_id,"observation_idx":s["obs"],
                "predicted_arc_id":s["arc_id"],"candidate_score":s["score"],
                "candidate_rank":s["rank"],"decoder":"xgboost_hmm_viterbi"})
        expanded,broken=expand_route(decoded,graph)
        pairs=list(zip(decoded[:-1],decoded[1:]))
        direct=sum((p["v"]==q["u"]) or (p["arc_id"]==q["arc_id"] and q["fraction"]>=p["fraction"]) for p,q in pairs)
        continuity=(len(pairs)-broken)/len(pairs) if pairs else 1.0
        direct_rate=direct/len(pairs) if pairs else 1.0
        route_outputs.append({"trip_id":trip_id,"expanded_arc_ids":expanded,"unbridgeable_transitions":broken})
        summaries.append({"trip_id":trip_id,"observations_decoded":len(decoded),
            "expanded_route_arcs":len(expanded),"transition_count":len(pairs),
            "reachable_transition_rate":continuity,"direct_join_rate":direct_rate,
            "unbridgeable_transitions":broken})

        # Evaluation-only section, run after predictions: derive route-aligned targets.
        route_path=Path(str(prefix)+".route")
        if route_path.exists():
            route_values=np.loadtxt(route_path,dtype=float,ndmin=1)
            targets=align_reference_route(track,nodes,arcs,route_values)
            for obs,target in enumerate(targets):
                eval_label_rows.append({"trip_id":trip_id,"observation_idx":obs,
                    "aligned_target_arc_id":int(target) if target>=0 else np.nan,
                    "route_aligned":bool(target>=0)})

    RESULTS.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(PRED_OUT,index=False,compression="gzip")
    pd.DataFrame(summaries).to_csv(SUMMARY_OUT,index=False)
    ROUTES_OUT.write_text(json.dumps(route_outputs,indent=2),encoding="utf-8")
    if eval_label_rows:
        labels=pd.DataFrame(eval_label_rows)
        labels.to_csv(LABELS_OUT,index=False,compression="gzip")
        # Compare classifiers and HMM only on evaluation-aligned student observations.
        lab=labels.loc[labels.route_aligned].copy()
        target=lab.set_index(["trip_id","observation_idx"]).aligned_target_arc_id
        all_groups=data.groupby(["trip_id","observation_idx"],sort=False)
        miss=[]; base_ok=[]; hmm_ok=[]; conditional=[]
        base_rows=data.loc[data.groupby(["trip_id","observation_idx"],sort=False).candidate_score.idxmax()]
        base_map={(r.trip_id,int(r.observation_idx)):int(r.candidate_arc_id) for r in base_rows.itertuples()}
        hmm_map={(r["trip_id"],int(r["observation_idx"])):int(r["predicted_arc_id"]) for r in prediction_rows}
        for key,targ in target.items():
            candidates=set(int(x) for x in all_groups.get_group(key).candidate_arc_id)
            hit=int(targ) in candidates; miss.append(not hit)
            b=base_map.get(key,-1); h=hmm_map.get(key,-1)
            base_ok.append(b==int(targ)); hmm_ok.append(h==int(targ))
            if hit: conditional.append((b==int(targ),h==int(targ)))
        metric={"partition":"student_testing","aligned_observations":len(miss),
            "candidate_miss_rate":float(np.mean(miss)),
            "classifier_end_to_end_top1":float(np.mean(base_ok)),
            "hmm_end_to_end_top1":float(np.mean(hmm_ok)),
            "classifier_top1_when_target_candidate_exists":float(np.mean([x[0] for x in conditional])) if conditional else None,
            "hmm_top1_when_target_candidate_exists":float(np.mean([x[1] for x in conditional])) if conditional else None}
        pd.DataFrame([metric]).to_csv(RESULTS/"hmm_viterbi_student_test_metrics.csv",index=False)
        print("Student-test eval-only metrics:",metric)
    print("Saved HMM predictions, per-trip summaries, and expanded route sequences.")
    print("Reference .route files were used only after decoding to create held-out eval labels.")

if __name__=="__main__": main()
