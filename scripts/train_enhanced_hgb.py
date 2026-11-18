"""
PathSnap Enhanced HistGradientBoosting Training & Evaluation Script.

Features:
  - 14 inference-safe features including relative candidate ranking:
    ['dist_rank', 'dist_gap', 'dist_ratio', 'distance_meters',
     'heading_cos', 'heading_diff_deg', 'heading_sin', 'projection_fraction',
     'segment_length_m', 'gps_speed_mps', 'step_distance_m', 'proximity_score',
     'from_node_in_degree', 'to_node_out_degree']
  - 5-Fold Group-Aware Cross-Validation (strictly grouped by trip_id)
  - Full evaluation on 10 student-test trips
  - Classifier-only vs Classifier + HMM route reconstruction
  - Artifact serialization:
    * artifacts/models/hist_gradient_boosting_enhanced.joblib
    * artifacts/models/hist_gradient_boosting_enhanced_metadata.json
    * results/accuracy_improvement/enhanced_hgb_metrics.csv
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import GroupKFold

from src.data.parsers import load_trip
from src.data.road_graph import RoadGraph
from src.data.alignment import align_reference_route_to_observations
from src.data.enhanced_features import (
    ENHANCED_FEATURE_NAMES,
    extract_enhanced_candidates,
)
from src.hmm.states import CandidateState
from src.hmm.viterbi import ViterbiDecoder
from src.hmm.decode_route import reconstruct_continuous_route
from src.hmm.route_metrics import calculate_route_continuity, calculate_top1_accuracy, calculate_candidate_miss_rate

REPO_ROOT = Path(__file__).resolve().parent.parent


def locate_partition(partition_name: str) -> Path:
    candidates = [
        REPO_ROOT.parent / "hack" / "Student folder" / "Student folder" / partition_name,
        REPO_ROOT.parent / "hack" / "Student folder" / partition_name,
        REPO_ROOT / "data" / "raw" / partition_name,
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*/*.track"))) >= 10:
            return c
    for c in candidates:
        if c.exists() and any(c.glob("*/*.track")):
            return c
    raise FileNotFoundError(f"{partition_name} directory not found")


def extract_trip_dataset(
    trip_dir: Path,
    include_labels: bool = True,
    search_radius_m: float = 75.0,
    max_cands: int = 12,
) -> Dict:
    trip = load_trip(trip_dir, include_route=include_labels)
    tid = trip_dir.name
    track = trip["track"]
    N = len(track)

    lons = track["longitude"].to_numpy(dtype=float)
    lats = track["latitude"].to_numpy(dtype=float)
    timestamps = track["timestamp"].to_numpy(dtype=float)

    rg = RoadGraph(
        trip["nodes"][["longitude", "latitude"]].to_numpy(dtype=float),
        trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int),
        trip_id=tid,
    )

    true_arcs = None
    if include_labels and trip["route"] is not None and len(trip["route"]) > 0:
        raw_route = trip["route"]["arc_id"].to_numpy(dtype=int)
        if len(raw_route) == N:
            true_arcs = raw_route
        else:
            true_arcs = align_reference_route_to_observations(
                lons, lats, timestamps, rg, raw_route
            )

    obs_cands, X_feat, row_to_obs = extract_enhanced_candidates(
        rg, lons, lats, timestamps, search_radius_m=search_radius_m, max_cands=max_cands
    )

    labels = []
    if true_arcs is not None:
        for t in range(N):
            gt_aid = true_arcs[t]
            for cand in obs_cands[t]:
                labels.append(1 if cand.arc_id == gt_aid else 0)
        y_labels = np.array(labels, dtype=np.int8)
    else:
        y_labels = None

    return {
        "trip_id": tid,
        "observations": N,
        "road_graph": rg,
        "lons": lons,
        "lats": lats,
        "timestamps": timestamps,
        "true_arcs": true_arcs,
        "obs_cands": obs_cands,
        "X_features": X_feat,
        "y_labels": y_labels,
        "row_to_obs": row_to_obs,
    }


def main():
    train_dir = locate_partition("train")
    test_dir = locate_partition("student_test")

    out_models = REPO_ROOT / "artifacts" / "models"
    out_results = REPO_ROOT / "results" / "accuracy_improvement"
    out_models.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("PATHSNAP ENHANCED HISTGRADIENTBOOSTING TRAINING & EVALUATION")
    print(f"Features ({len(ENHANCED_FEATURE_NAMES)}): {ENHANCED_FEATURE_NAMES}")
    print("=" * 85)

    # 1. Load Training Data
    train_folders = sorted([f for f in train_dir.iterdir() if f.is_dir() and (f / f"{f.name}.track").exists()])
    print(f"\n[1/5] Extracting candidate features from {len(train_folders)} training trips...")
    
    t0_extract = time.time()
    train_trips_data = []
    all_X_list = []
    all_y_list = []
    all_groups_list = []

    for idx, f in enumerate(train_folders):
        data = extract_trip_dataset(f, include_labels=True, search_radius_m=75.0, max_cands=12)
        train_trips_data.append(data)
        all_X_list.append(data["X_features"])
        all_y_list.append(data["y_labels"])
        all_groups_list.extend([data["trip_id"]] * len(data["y_labels"]))
        if (idx + 1) % 20 == 0 or (idx + 1) == len(train_folders):
            print(f"  Processed {idx + 1}/{len(train_folders)} training trips ({time.time() - t0_extract:.1f}s)")

    X_train_all = np.vstack(all_X_list)
    y_train_all = np.concatenate(all_y_list)
    groups_all = np.array(all_groups_list)

    print(f"Total training candidate samples: {len(y_train_all):,}")
    print(f"Positive matches (y=1): {int(np.sum(y_train_all)):,} ({np.mean(y_train_all)*100:.2f}%)")

    # 2. 5-Fold GroupKFold Cross-Validation
    print("\n[2/5] Running 5-Fold Group-Aware Cross-Validation (strictly grouped by trip_id)...")
    gkf = GroupKFold(n_splits=5)
    fold_metrics = []

    # Map trip_ids to their index in train_trips_data
    trip_id_to_data = {d["trip_id"]: d for d in train_trips_data}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_all, y_train_all, groups=groups_all), start=1):
        t0_fold = time.time()
        X_tr, y_tr = X_train_all[train_idx], y_train_all[train_idx]
        X_va, y_val = X_train_all[val_idx], y_train_all[val_idx]
        val_groups = set(groups_all[val_idx])

        clf_fold = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=63,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42 + fold,
        )
        clf_fold.fit(X_tr, y_tr)

        val_scores = clf_fold.predict_proba(X_va)[:, 1]
        val_preds = (val_scores >= 0.5).astype(int)

        f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        loss = log_loss(y_val, np.clip(val_scores, 1e-7, 1 - 1e-7))

        # Observation-level Top-1 accuracy across validation trips
        val_top1_list = []
        val_miss_list = []
        val_cont_list = []

        for v_tid in sorted(val_groups):
            v_data = trip_id_to_data[v_tid]
            v_scores = clf_fold.predict_proba(v_data["X_features"])[:, 1]
            v_cands = v_data["obs_cands"]
            v_gt = v_data["true_arcs"]
            N_v = v_data["observations"]

            # Argmax prediction per observation
            v_pred_arcs = []
            idx_sc = 0
            for t in range(N_v):
                cnt = len(v_cands[t])
                sc = v_scores[idx_sc : idx_sc + cnt]
                idx_sc += cnt
                if cnt > 0:
                    v_pred_arcs.append(v_cands[t][int(np.argmax(sc))].arc_id)
                else:
                    v_pred_arcs.append(-1)
            v_pred_arcs = np.array(v_pred_arcs)

            t1 = calculate_top1_accuracy(v_pred_arcs, v_gt)
            miss = calculate_candidate_miss_rate(v_cands, v_gt)
            cont = calculate_route_continuity(v_pred_arcs, v_data["road_graph"])

            val_top1_list.append(t1)
            val_miss_list.append(miss)
            val_cont_list.append(cont)

        mean_top1 = float(np.mean(val_top1_list))
        mean_miss = float(np.mean(val_miss_list))
        mean_cont = float(np.mean(val_cont_list))

        fold_res = {
            "fold": fold,
            "val_trips": len(val_groups),
            "macro_f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "log_loss": round(loss, 4),
            "top1_accuracy": round(mean_top1, 4),
            "candidate_miss_rate": round(mean_miss, 4),
            "route_continuity": round(mean_cont, 4),
            "train_time_s": round(time.time() - t0_fold, 1),
        }
        fold_metrics.append(fold_res)
        print(f"  Fold {fold} | Macro F1: {f1:.4f} | Top-1 Acc: {mean_top1*100:>5.2f}% | "
              f"Miss Rate: {mean_miss*100:>4.2f}% | Continuity: {mean_cont*100:>5.2f}% | ({fold_res['train_time_s']}s)")

    df_folds = pd.DataFrame(fold_metrics)
    print("\n--- 5-FOLD CROSS-VALIDATION SUMMARY ---")
    print(f"CV Macro F1:         {df_folds['macro_f1'].mean():.4f} +/- {df_folds['macro_f1'].std():.4f}")
    print(f"CV Top-1 Accuracy:   {df_folds['top1_accuracy'].mean()*100:.2f}% +/- {df_folds['top1_accuracy'].std()*100:.2f}%")
    print(f"CV Route Continuity: {df_folds['route_continuity'].mean()*100:.2f}%")
    print(f"CV Candidate Miss:   {df_folds['candidate_miss_rate'].mean()*100:.2f}%")

    # 3. Fit Final Model on 100% of Training Data
    print("\n[3/5] Fitting final model on all 80 training trips...")
    t0_fit = time.time()
    final_model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=63,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(X_train_all, y_train_all)
    fit_time = time.time() - t0_fit
    print(f"Final model fit completed in {fit_time:.2f} seconds.")

    # Save artifacts
    model_save_path = out_models / "hist_gradient_boosting_enhanced.joblib"
    meta_save_path = out_models / "hist_gradient_boosting_enhanced_metadata.json"
    joblib.dump(final_model, model_save_path)
    with open(meta_save_path, "w") as fp:
        json.dump({
            "model_type": "HistGradientBoostingClassifier",
            "features": ENHANCED_FEATURE_NAMES,
            "params": final_model.get_params(),
            "cv_metrics": {
                "macro_f1_mean": float(df_folds["macro_f1"].mean()),
                "top1_accuracy_mean": float(df_folds["top1_accuracy"].mean()),
                "route_continuity_mean": float(df_folds["route_continuity"].mean()),
            },
            "training_samples": len(y_train_all),
            "training_trips": len(train_folders),
            "training_time_s": fit_time,
        }, fp, indent=2, default=str)
    print(f"Saved model to: {model_save_path}")

    # 4. Evaluate on Student-Testing Trips
    print("\n[4/5] Evaluating on all 10 student-test trips...")
    test_folders = sorted([f for f in test_dir.iterdir() if f.is_dir() and (f / f"{f.name}.track").exists()])

    test_results = []
    for f in test_folders:
        tid = f.name
        t_data = extract_trip_dataset(f, include_labels=True, search_radius_m=75.0, max_cands=12)
        rg = t_data["road_graph"]
        lons = t_data["lons"]
        lats = t_data["lats"]
        timestamps = t_data["timestamps"]
        true_arcs = t_data["true_arcs"]
        obs_cands = t_data["obs_cands"]
        N = t_data["observations"]

        t_inf_0 = time.perf_counter()
        cand_scores = final_model.predict_proba(t_data["X_features"])[:, 1]
        t_inf_ms = (time.perf_counter() - t_inf_0) * 1000

        scores_per_obs = []
        idx = 0
        for t in range(N):
            cnt = len(obs_cands[t])
            scores_per_obs.append(cand_scores[idx : idx + cnt])
            idx += cnt

        # Classifier-only
        clf_arcs = []
        for t in range(N):
            cands = obs_cands[t]
            sc = scores_per_obs[t]
            if len(cands) > 0 and len(sc) > 0:
                clf_arcs.append(cands[int(np.argmax(sc))].arc_id)
            else:
                clf_arcs.append(-1)
        clf_arcs = np.array(clf_arcs)

        # HMM Viterbi
        t_dec_0 = time.perf_counter()
        decoder = ViterbiDecoder(
            road_graph=rg,
            beta_meters=18.0,
            sigma_z_meters=10.0,
            max_speed_mps=45.0,
            gap_threshold_sec=10.0,
        )
        _, hmm_arcs_list, _, _ = decoder.decode(obs_cands, scores_per_obs, lons, lats, timestamps)
        hmm_arcs = np.array(hmm_arcs_list)
        t_dec_ms = (time.perf_counter() - t_dec_0) * 1000

        clf_acc = calculate_top1_accuracy(clf_arcs, true_arcs)
        hmm_acc = calculate_top1_accuracy(hmm_arcs, true_arcs)
        miss_rate = calculate_candidate_miss_rate(obs_cands, true_arcs)
        clf_cont = calculate_route_continuity(clf_arcs, rg)
        hmm_cont = calculate_route_continuity(hmm_arcs, rg)

        rec = {
            "trip_id": tid,
            "observations": N,
            "candidates": len(t_data["X_features"]),
            "candidate_miss_rate": round(float(miss_rate), 4),
            "clf_top1_accuracy": round(float(clf_acc), 4),
            "hmm_top1_accuracy": round(float(hmm_acc), 4),
            "acc_delta": round(float(hmm_acc - clf_acc), 4),
            "clf_continuity": round(float(clf_cont), 4),
            "hmm_continuity": round(float(hmm_cont), 4),
            "cont_delta": round(float(hmm_cont - clf_cont), 4),
            "clf_latency_ms": round(float(t_inf_ms), 1),
            "hmm_latency_ms": round(float(t_inf_ms + t_dec_ms), 1),
        }
        test_results.append(rec)
        print(f"Trip {tid:<8} | Miss: {miss_rate*100:>4.1f}% | Acc: {clf_acc*100:>5.1f}% -> {hmm_acc*100:>5.1f}% "
              f"({rec['acc_delta']*100:>+4.1f}%) | Cont: {clf_cont*100:>5.1f}% -> {hmm_cont*100:>5.1f}%")

    df_test = pd.DataFrame(test_results)
    test_csv = out_results / "enhanced_hgb_student_test.csv"
    df_test.to_csv(test_csv, index=False)
    print(f"\nSaved student-test results to {test_csv}")

    print("\n" + "=" * 85)
    print("--- FINAL ENHANCED MODEL STUDENT-TEST PERFORMANCE ---")
    print(f"Macro Top-1 Accuracy (Classifier-Only): {df_test['clf_top1_accuracy'].mean()*100:.2f}%")
    print(f"Macro Top-1 Accuracy (Classifier + HMM): {df_test['hmm_top1_accuracy'].mean()*100:.2f}%")
    print(f"Macro Route Continuity (Classifier + HMM): {df_test['hmm_continuity'].mean()*100:.2f}%")
    print(f"Macro Candidate Miss Rate:               {df_test['candidate_miss_rate'].mean()*100:.2f}%")
    print("=" * 85)


if __name__ == "__main__":
    main()
