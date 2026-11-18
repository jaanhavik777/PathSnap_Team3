"""Train a reproducible MLP and evaluate with trip-grouped validation."""
from __future__ import annotations
import json,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,log_loss,confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.person4_candidates import FEATURE_COLUMNS,training_frame,read_trip,candidate_features

def make_model(hidden=(64,32),alpha=.001,seed=42,max_iter=250):
    return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),
        ("mlp",MLPClassifier(hidden_layer_sizes=tuple(hidden),activation="relu",solver="adam",alpha=alpha,
         batch_size="auto",learning_rate_init=.001,max_iter=max_iter,early_stopping=True,
         validation_fraction=.15,n_iter_no_change=20,random_state=seed))])

def grouped_cv(frame,config,folds=5):
    splitter=GroupKFold(n_splits=min(folds,frame.trip_id.nunique())); scores=[]; train_scores=[]
    for tr,va in splitter.split(frame[FEATURE_COLUMNS],frame.label,frame.trip_id):
        model=make_model(**config); model.fit(frame.iloc[tr][FEATURE_COLUMNS],frame.iloc[tr].label)
        scores.append(f1_score(frame.iloc[va].label,model.predict(frame.iloc[va][FEATURE_COLUMNS]),average="macro",zero_division=0))
        train_scores.append(f1_score(frame.iloc[tr].label,model.predict(frame.iloc[tr][FEATURE_COLUMNS]),average="macro",zero_division=0))
    return {"cv_macro_f1_mean":float(np.mean(scores)),"cv_macro_f1_std":float(np.std(scores)),"cv_train_macro_f1_mean":float(np.mean(train_scores)),"fold_scores":scores}

def evaluate_candidates(model,frame):
    frame=frame[frame.reference_arc_id>=0].copy()
    if frame.empty:return {"status":"unavailable_no_labeled_candidates"}
    X=frame[FEATURE_COLUMNS]; y=frame.label.to_numpy(); pred=model.predict(X); prob=model.predict_proba(X)[:,1]; hits=total=0
    scored=frame.assign(_score=prob)
    for _,g in scored.groupby(["trip_id","observation_id"]):
        ref=int(g.reference_arc_id.iloc[0])
        if ref>=0: total+=1; hits+=int(g.loc[g._score.idxmax(),"candidate_arc_id"]==ref)
    return {"candidate_accuracy":float(accuracy_score(y,pred)),"candidate_macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
     "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
     "log_loss":float(log_loss(y,np.c_[1-prob,prob],labels=[0,1])),"observation_top1_accuracy":hits/total if total else None,
     "labeled_observations":total,"candidate_rows":len(frame),"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def run(data_root,out_root,seed=42,candidate_count=4):
    out=Path(out_root); out.mkdir(parents=True,exist_ok=True); frame=training_frame(data_root,candidate_count)
    # Keep the demo experiment reproducible and tractable on a laptop; early
    # stopping exits sooner when the held-out trip-fold score stops improving.
    config={"hidden":(32,),"alpha":.001,"seed":seed,"max_iter":80}; cv=grouped_cv(frame,config,folds=3)
    model=make_model(**config); t0=time.perf_counter(); model.fit(frame[FEATURE_COLUMNS],frame.label); train_s=time.perf_counter()-t0
    student=[]
    for folder in sorted((Path(data_root)/"student_test").iterdir()):
        if folder.is_dir():
            f=candidate_features(read_trip(folder),candidate_count=candidate_count)
            if not f.empty and f.reference_arc_id.ge(0).any():student.append(f)
    sf=pd.concat(student,ignore_index=True) if student else pd.DataFrame()
    t0=time.perf_counter(); metrics=evaluate_candidates(model,sf); infer_s=time.perf_counter()-t0
    joblib.dump({"model":model,"feature_columns":FEATURE_COLUMNS,"config":config,"seed":seed,"candidate_count":candidate_count},out/"mlp_pipeline.joblib")
    summary={"model":"MLPClassifier","config":config,"feature_columns":FEATURE_COLUMNS,"seed":seed,"candidate_count":candidate_count,"training_rows":len(frame),"training_trips":int(frame.trip_id.nunique()),"positive_rows":int(frame.label.sum()),"cv":cv,"student_test":metrics,"training_time_s":train_s,"student_inference_time_s":infer_s,"student_trips_evaluated":int(sf.trip_id.nunique()) if not sf.empty else 0,"training_iterations":int(model.named_steps["mlp"].n_iter_),"training_converged_before_limit":bool(model.named_steps["mlp"].n_iter_<config["max_iter"]),"training_loss_curve":[float(x) for x in model.named_steps["mlp"].loss_curve_],"labeling_note":"Reference route arc set matched to nearest candidate per GPS point only within distance and ambiguity thresholds; all candidate rows for unlabeled or ambiguous observations are excluded rather than encoded as negative."}
    (out/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    pd.DataFrame([{"configuration":"32 relu alpha=.001",**cv,**{f"student_{k}":v for k,v in metrics.items() if not isinstance(v,(list,dict))},"training_time_s":train_s,"student_inference_time_s":infer_s}]).to_csv(out/"metrics.csv",index=False)
    _plots(summary,out); return summary

def _plots(summary,out):
    import matplotlib.pyplot as plt
    m=summary["student_test"]
    if summary["training_loss_curve"]:
        plt.figure();plt.plot(summary["training_loss_curve"]);plt.xlabel("Iteration");plt.ylabel("Training loss");plt.title("MLP training loss");plt.tight_layout();plt.savefig(out/"training_loss.png",dpi=150);plt.close()
    if "confusion_matrix" in m:
        plt.figure();plt.imshow(m["confusion_matrix"],cmap="Blues");plt.xticks([0,1],["non-match","match"]);plt.yticks([0,1],["non-match","match"]);plt.xlabel("Predicted");plt.ylabel("Reference");plt.title("Student-test candidate confusion matrix");plt.colorbar();plt.tight_layout();plt.savefig(out/"confusion_matrix.png",dpi=150);plt.close()
    vals=[summary["cv"]["cv_macro_f1_mean"],m.get("candidate_macro_f1",0) or 0,m.get("observation_top1_accuracy",0) or 0]
    plt.figure();plt.bar(["Grouped CV macro F1","Student candidate macro F1","Student observation Top-1"],vals,color=["#5969d9","#11a579","#f2a541"]);plt.ylim(0,1);plt.xticks(rotation=15,ha="right");plt.tight_layout();plt.savefig(out/"performance.png",dpi=150);plt.close()
