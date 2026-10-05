"""
Prediction Engine & Inference Service
Project: In-Game Cheat Detection System
File: backend/prediction.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'ml', 'model', 'cheat_detection_model.pkl'))
PIPELINE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'ml', 'model', 'preprocessing_pipeline.pkl'))
METADATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'ml', 'model', 'metadata.json'))

_model = None
_pipeline = None
_metadata = None

FEATURE_COLUMNS = [
    'yaw', 'pitch', 'delta_yaw', 'delta_pitch',
    'accel_yaw', 'accel_pitch', 'target_x', 'target_y', 'target_z',
    'position_x', 'position_y', 'position_z', 'sensitivity', 'new_sequence'
]

def load_artifacts():
    global _model, _pipeline, _metadata
    if _model is not None and _pipeline is not None:
        return _model, _pipeline

    if os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            _pipeline = joblib.load(PIPELINE_PATH)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    _metadata = json.load(f)
            return _model, _pipeline
        except Exception as e:
            print(f"Artifact loading note: {e}")

    return None, None

def evaluate_telemetry(features_dict_or_list):
    """
    Accepts 14 telemetry features (either as a list or dict)
    Transforms through scikit-learn Pipeline and evaluates model probability.
    """
    model, pipeline = load_artifacts()

    if isinstance(features_dict_or_list, dict):
        raw_list = [float(features_dict_or_list[k]) for k in FEATURE_COLUMNS]
    else:
        raw_list = [float(x) for x in features_dict_or_list]

    delta_yaw = raw_list[2]
    delta_pitch = raw_list[3]
    accel_yaw = raw_list[4]
    accel_pitch = raw_list[5]
    sensitivity = raw_list[12]

    if model is not None and pipeline is not None:
        try:
            # Transform through pipeline
            transformed = pipeline.transform([raw_list])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(transformed)[0][1]) * 100.0
            else:
                prob = float(model.decision_function(transformed)[0])
                prob = 100.0 / (1.0 + np.exp(-prob))
        except Exception as e:
            prob = 50.0
    else:
        # Calibrated heuristic fallback matching dataset boundaries
        # Legit: delta_yaw ~ 10.06, accel_yaw ~ 10.69
        # Cheat: delta_yaw ~ 25.36, accel_yaw ~ 33.76
        score = 0.0
        abs_dy = abs(delta_yaw)
        abs_ay = abs(accel_yaw)
        if abs_dy > 30.0: score += 45.0
        elif abs_dy > 18.0: score += 25.0
        elif abs_dy > 10.0: score += 10.0

        if abs_ay > 40.0: score += 45.0
        elif abs_ay > 20.0: score += 25.0
        elif abs_ay > 10.0: score += 10.0

        prob = min(99.5, max(1.5, score))

    # Stage 6 Threshold Rules:
    # 0 - 30%    : LOW RISK (Legitimate)
    # 31 - 70%   : MEDIUM RISK (Suspicious / Flagged)
    # 71 - 100%  : HIGH RISK (Suspicious / Critical)
    alert_threshold = 31.0
    pred_label = "SUSPICIOUS" if prob >= alert_threshold else "LEGITIMATE"

    if prob <= 30.0:
        risk = "LOW"
    elif prob <= 70.0:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "probability": round(prob, 2),
        "prediction": pred_label,
        "risk_level": risk,
        "alert_threshold": alert_threshold,
        "metrics": {
            "delta_yaw": abs(delta_yaw),
            "delta_pitch": abs(delta_pitch),
            "accel_yaw": abs(accel_yaw),
            "accel_pitch": abs(accel_pitch),
            "sensitivity": sensitivity
        }
    }
