"""
Stages 4, 5, & 6: Feature Selection, Model Training, Benchmarking, & Risk Export
Project: In-Game Cheat Detection System
File: ml/train_and_evaluate.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    from ml.stage3_preprocessing import perform_group_aware_split
except ImportError:
    from stage3_preprocessing import perform_group_aware_split

def execute_training_and_evaluation(dataset_path="dataset/final_dataset.csv"):
    print("=" * 80, flush=True)
    print(" STAGE 4: FEATURE SELECTION (TRAINING DATA ONLY)", flush=True)
    print("=" * 80, flush=True)

    # 1. Perform Group-Aware Split
    split_data = perform_group_aware_split(dataset_path=dataset_path, group_col='uuid', test_size=0.20, random_state=42)
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    pipeline = split_data['pipeline']

    feature_names = pipeline.named_steps['transformer'].feature_names_
    print(f"\nExtracted Feature Space ({len(feature_names)} features):", flush=True)
    print(", ".join(feature_names), flush=True)

    # 2. Fast Mutual Information on Training Sample
    print("\nComputing Mutual Information on Training Split...", flush=True)
    np.random.seed(42)
    sample_size = min(2000, len(X_train))
    sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
    mi_scores = mutual_info_classif(X_train[sample_indices], y_train[sample_indices], random_state=42)
    
    mi_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Info': mi_scores
    }).sort_values(by='Mutual_Info', ascending=False)

    print("\nTop Features by Mutual Information (Label Signal):", flush=True)
    print("-" * 55, flush=True)
    for idx, row in mi_ranking.head(10).iterrows():
        print(f"  {row['Feature']:<20}: MI = {row['Mutual_Info']:.4f}", flush=True)
    print("-" * 55, flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" STAGE 5: MODEL TRAINING & BENCHMARKING (GROUP-AWARE TEST SET)", flush=True)
    print("=" * 80, flush=True)

    # Candidate Classifier Setup
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, 
            solver='lbfgs', 
            max_iter=500, 
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, 
            min_samples_split=20, 
            min_samples_leaf=10, 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, 
            max_depth=10, 
            min_samples_split=15, 
            min_samples_leaf=5, 
            n_jobs=-1,
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=9, 
            weights='distance',
            n_jobs=-1
        )
    }

    benchmark_records = []
    trained_estimators = {}

    for name, clf in candidate_models.items():
        print(f"Training [{name}]...", flush=True)
        clf.fit(X_train, y_train)
        trained_estimators[name] = clf

        preds = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        else:
            probs = clf.decision_function(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)

        benchmark_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc,
            "CM": cm
        })

    # Benchmark Table
    summary_df = pd.DataFrame(benchmark_records)
    print("\n" + "=" * 80, flush=True)
    print(" BENCHMARK PERFORMANCE TABLE (EVALUATED ON UNSEEN PLAYERS)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'ROC-AUC':<10}", flush=True)
    print("-" * 80, flush=True)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<22} | {row['Accuracy']*100:>8.2f}% | {row['Precision']*100:>8.2f}% | {row['Recall']*100:>8.2f}% | {row['F1']*100:>8.2f}% | {row['ROC-AUC']:>10.4f}", flush=True)
    print("=" * 80, flush=True)

    # Best Model Selection
    best_row = summary_df.sort_values(by=['ROC-AUC', 'F1'], ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_clf = trained_estimators[best_model_name]

    print("\n" + "=" * 80, flush=True)
    print(f" SELECTED BEST PRODUCTION MODEL: {best_model_name}", flush=True)
    print(f" ROC-AUC: {best_row['ROC-AUC']:.4f} | F1-Score: {best_row['F1']*100:.2f}% | Accuracy: {best_row['Accuracy']*100:.2f}%", flush=True)
    print("=" * 80, flush=True)

    # Stage 6: Persist Artifacts
    model_dir = "ml/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "cheat_detection_model.pkl")
    joblib.dump(best_clf, model_path)
    print(f">> Persisted Best Model to: {model_path}", flush=True)

    metadata = {
        "model_version": "v2.0-group-aware",
        "selected_algorithm": best_model_name,
        "metrics": {
            "accuracy": float(best_row['Accuracy']),
            "precision": float(best_row['Precision']),
            "recall": float(best_row['Recall']),
            "f1_score": float(best_row['F1']),
            "roc_auc": float(best_row['ROC-AUC'])
        },
        "thresholds": {
            "suspicious_alert_threshold": 31.0,
            "low_risk_max": 30.0,
            "medium_risk_max": 70.0,
            "high_risk_min": 71.0
        },
        "feature_list": feature_names,
        "input_features_required": [
            'yaw', 'pitch', 'delta_yaw', 'delta_pitch',
            'accel_yaw', 'accel_pitch', 'target_x', 'target_y', 'target_z',
            'position_x', 'position_y', 'position_z', 'sensitivity', 'new_sequence'
        ]
    }

    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f">> Persisted Metadata to: {metadata_path}", flush=True)

    return {
        "summary_df": summary_df,
        "best_model_name": best_model_name,
        "metadata": metadata
    }

if __name__ == '__main__':
    execute_training_and_evaluation()
