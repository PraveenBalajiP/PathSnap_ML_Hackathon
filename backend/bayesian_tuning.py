"""Bayesian tuning for PathSnap's XGBoost candidate scorer.
Objective: mean observation-level top-1 accuracy in GroupKFold split by trip.
"""
from pathlib import Path
import argparse, json, time
import joblib, numpy as np, optuna, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets"/"processed_v1"; OUT=ROOT/"datasets"/"model_results"
NUMERIC=["gps_step_m","gps_speed_mps","time_delta_s","candidate_rank","candidate_distance_m","projection_fraction","arc_length_m","from_node_out_degree","from_node_in_degree","to_node_out_degree","to_node_in_degree"]
ANGLES=["gps_heading_deg","gps_heading_change_deg","arc_bearing_deg","heading_difference_deg"]
SEED=42

def features(df):
    x=df[NUMERIC].copy()
    for c in ANGLES:
        r=np.deg2rad(pd.to_numeric(df[c],errors="coerce")); x[c+"_sin"],x[c+"_cos"]=np.sin(r),np.cos(r)
    return x.replace([np.inf,-np.inf],np.nan).astype("float32")

def top1(df,scores):
    d=df[["trip_id","observation_idx","is_target"]].copy(); d["score"]=scores
    best=d.loc[d.groupby(["trip_id","observation_idx"],sort=False).score.idxmax()]
    return float(best.is_target.mean())

def model(params,y,device):
    pos=max(int((y==1).sum()),1); neg=int((y==0).sum())
    clf=XGBClassifier(**params,scale_pos_weight=neg/pos,objective="binary:logistic",
        eval_metric="logloss",tree_method="hist",device=device,n_jobs=-1,random_state=SEED)
    return make_pipeline(SimpleImputer(strategy="median",add_indicator=True),clf)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trials",type=int,default=12)
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--timeout",type=int,default=None)
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda",
        help="Training device; CUDA is the default when an NVIDIA GPU is available.")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(DATA/"training_candidates.csv",dtype={"trip_id":str})
    student=pd.read_csv(DATA/"student_test_candidates.csv",dtype={"trip_id":str})
    train["is_target"]=train["is_target"].astype(np.int8)
    X=features(train); y=train["is_target"].to_numpy(); groups=train["trip_id"].to_numpy()
    cv=list(GroupKFold(n_splits=args.folds).split(X,y,groups))
    print(f"Using {len(train):,} rows from {train.trip_id.nunique()} trips; {len(student):,} student candidates.")
    study=optuna.create_study(study_name=f"pathsnap_xgb_group_top1_{args.folds}fold",
        storage=f"sqlite:///{(OUT/'xgboost_bayesian_study.db').as_posix()}",
        direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED,multivariate=True),
        load_if_exists=True)
    def objective(t):
        p={"n_estimators":t.suggest_int("n_estimators",150,500,step=50),
           "max_depth":t.suggest_int("max_depth",4,10),
           "learning_rate":t.suggest_float("learning_rate",.02,.20,log=True),
           "min_child_weight":t.suggest_float("min_child_weight",1,20,log=True),
           "subsample":t.suggest_float("subsample",.65,1),
           "colsample_bytree":t.suggest_float("colsample_bytree",.60,1),
           "reg_lambda":t.suggest_float("reg_lambda",.1,20,log=True),
           "gamma":t.suggest_float("gamma",0,5)}
        tops,aps,times=[],[],[]
        for fold,(a,z) in enumerate(cv,1):
            clf=model(p,y[a],args.device); started=time.perf_counter(); clf.fit(X.iloc[a],y[a]); times.append(time.perf_counter()-started)
            prob=clf.predict_proba(X.iloc[z])[:,1]
            tops.append(top1(train.iloc[z],prob)); aps.append(average_precision_score(y[z],prob))
            t.report(float(np.mean(tops)),step=fold)
            if t.should_prune(): raise optuna.TrialPruned()
        t.set_user_attr("fold_top1",tops); t.set_user_attr("fold_average_precision",aps); t.set_user_attr("fold_fit_seconds",times)
        return float(np.mean(tops))
    study.optimize(objective,n_trials=args.trials,timeout=args.timeout,gc_after_trial=True)
    best=study.best_trial
    study.trials_dataframe(attrs=("number","value","params","state","user_attrs")).to_csv(OUT/"xgboost_bayesian_trials.csv",index=False)
    report={"model":"XGBoost","device":args.device,
        "selection_reason":"Highest mean grouped-CV observation top-1; average precision is reported as a secondary metric.",
        "objective":"mean observation-level top-1; GroupKFold by trip_id",
        "best_trial":best.number,"best_cv_top1_mean":best.value,"fold_top1":best.user_attrs.get("fold_top1"),
        "fold_average_precision":best.user_attrs.get("fold_average_precision"),
        "fold_fit_seconds":best.user_attrs.get("fold_fit_seconds"),"best_parameters":best.params,
        "trials":len(study.trials),"training_rows":len(train),"training_trips":train.trip_id.nunique(),
        "student_candidate_rows":len(student),"folds":args.folds,"seed":SEED}
    final=model(best.params,y,args.device); started=time.perf_counter(); final.fit(X,y)
    report["full_fit_seconds"]=time.perf_counter()-started
    joblib.dump(final,OUT/"xgboost_tuned_model.joblib")
    probability=final.predict_proba(features(student))[:,1]
    pred=student[["trip_id","observation_idx","candidate_arc_id","candidate_rank","candidate_distance_m"]].copy()
    pred["model"]="XGBoost Bayesian-tuned"; pred["candidate_score"]=probability; pred["predicted_top1"]=0
    ix=pred.groupby(["trip_id","observation_idx"],sort=False).candidate_score.idxmax(); pred.loc[ix,"predicted_top1"]=1
    pred.to_csv(OUT/"xgboost_tuned_student_test_candidate_scores.csv.gz",index=False,compression="gzip")
    report["student_test_candidate_rows_scored"]=len(pred)
    (OUT/"xgboost_bayesian_best.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Best grouped-CV top-1",round(best.value,4),"; saved tuned model and student candidate scores.")
    print("Student-test accuracy is not computed until separate eval-only labels are available.")

if __name__=="__main__": main()
