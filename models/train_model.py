"""
train_model.py
---------------
Objective 2 from the project proposal:
    "Train a Random Forest or XGBoost classifier as the primary model ...
     A small MLP or 1D-CNN can be added afterwards as a stretch
     comparison only if time remains."

Trains TWO classifiers on data/processed/{train,test}.csv:

  1. A "full" model on all 19 CICFlowMeter-style features -- this is the
     thorough offline evaluation Objective 2 asks for, and the numbers
     reported in the project report come from this model.
       -> models/artifacts/full/model.joblib, scaler.joblib, metrics.json,
          confusion_matrix.png, feature_importance.png

  2. A "live" model trained on only the 6 features a Ryu controller can
     actually compute from an OpenFlow FlowStatsReply at line rate
     (duration, packet/byte counts, derived rates, average packet size --
     see shared/mitigation_logic.LIVE_COMPUTABLE). This is a standard
     "heavy model for offline analysis, light model for low-latency
     inline inference" split: training directly on the reduced feature
     set beats training on all 19 and imputing the ones a live poll can't
     see, because the imputed values would otherwise drown out the real
     (if smaller) live signal. shared/mitigation_logic.py and
     controller/ryu_mitigation_app.py use THIS model.
       -> models/artifacts/live/model.joblib, scaler.joblib, metrics.json

Both use the same train/test split and the same RandomForest-vs-XGBoost
comparison; whichever scores higher F1 is kept for each.

USAGE
    python3 train_model.py
    python3 train_model.py --train ../data/processed/train.csv --test ../data/processed/test.csv
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

LABEL_COL = "label"
HERE = os.path.dirname(os.path.abspath(__file__))

# Must match shared/mitigation_logic.LIVE_COMPUTABLE exactly.
LIVE_FEATURES = [
    "flow_duration", "tot_fwd_pkts", "totlen_fwd_pkts",
    "flow_bytes_per_s", "flow_pkts_per_s", "avg_pkt_size",
]


def evaluate(name, model, X_test, y_test):
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_ms = (time.time() - t0) * 1000 / len(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_inference_ms_per_flow": infer_ms,
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    print(f"  [{name}] accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}  "
          f"({infer_ms:.4f} ms/flow)")
    print(f"  [{name}] confusion matrix [[TN FP] [FN TP]] = {cm}")
    return metrics, cm, y_pred


def train_and_select(X_train, y_train, X_test, y_test):
    """Train RF + (optionally) XGBoost, return (best_name, best_model, results_dict)."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results, models = {}, {}

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    t0 = time.time()
    rf.fit(X_train_s, y_train)
    m, cm, _ = evaluate("RandomForest", rf, X_test_s, y_test)
    m["train_seconds"] = time.time() - t0
    results["RandomForest"] = {"metrics": m, "confusion_matrix": cm}
    models["RandomForest"] = rf

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            n_jobs=-1, random_state=42,
        )
        t0 = time.time()
        xgb.fit(X_train_s, y_train)
        m, cm, _ = evaluate("XGBoost", xgb, X_test_s, y_test)
        m["train_seconds"] = time.time() - t0
        results["XGBoost"] = {"metrics": m, "confusion_matrix": cm}
        models["XGBoost"] = xgb
    else:
        print("  [train_model] xgboost not installed, skipping (Random Forest only).")

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1"])
    return best_name, models[best_name], scaler, results


def save_bundle(out_dir, features, best_name, best_model, scaler, results):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    with open(os.path.join(out_dir, "feature_list.json"), "w") as f:
        json.dump({"features": features}, f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    best_cm = np.array(results[best_name]["confusion_matrix"])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.title(f"{best_name} — Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=features).sort_values()
        plt.figure(figsize=(7, max(3, 0.35 * len(features))))
        imp.plot.barh(color="#2E86AB")
        plt.title(f"{best_name} — Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(HERE, "..", "data", "processed", "train.csv"))
    ap.add_argument("--test", default=os.path.join(HERE, "..", "data", "processed", "test.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts"))
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    all_features = [c for c in train_df.columns if c != LABEL_COL]

    # ---------------- 1. FULL model (Objective 2: thorough offline eval) --
    print(f"\n=== Training FULL model ({len(all_features)} features) ===")
    best_name, best_model, scaler, results = train_and_select(
        train_df[all_features], train_df[LABEL_COL],
        test_df[all_features], test_df[LABEL_COL],
    )
    print(f"  Best by F1: {best_name}")
    save_bundle(os.path.join(args.out, "full"), all_features, best_name, best_model, scaler, results)

    # ---------------- 2. LIVE model (Objective 4: inline controller) ------
    live_feats = [f for f in LIVE_FEATURES if f in all_features]
    print(f"\n=== Training LIVE model ({len(live_feats)} OpenFlow-derivable features) ===")
    best_name_l, best_model_l, scaler_l, results_l = train_and_select(
        train_df[live_feats], train_df[LABEL_COL],
        test_df[live_feats], test_df[LABEL_COL],
    )
    print(f"  Best by F1: {best_name_l}")
    save_bundle(os.path.join(args.out, "live"), live_feats, best_name_l, best_model_l, scaler_l, results_l)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({
            "full_model": {"name": best_name, "n_features": len(all_features),
                            "f1": results[best_name]["metrics"]["f1"],
                            "accuracy": results[best_name]["metrics"]["accuracy"]},
            "live_model": {"name": best_name_l, "n_features": len(live_feats),
                            "f1": results_l[best_name_l]["metrics"]["f1"],
                            "accuracy": results_l[best_name_l]["metrics"]["accuracy"]},
        }, f, indent=2)

    print(f"\n[train_model] artifacts written to {args.out}/full and {args.out}/live")


if __name__ == "__main__":
    main()
