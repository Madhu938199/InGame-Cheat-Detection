"""
Stage 3: Production Preprocessing Pipeline & Group-Aware Split
Project: In-Game Cheat Detection System
File: ml/stage3_preprocessing.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit

# Custom Transformer to handle raw telemetry cleaning and feature transformations
class TelemetryFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms raw telemetry inputs into clean numerical features:
    1. Fixes negative sensitivity values using absolute value and sensible clamping.
    2. Converts boolean new_sequence to integer (0/1).
    3. Wraps unbounded yaw into standard game orientation space [-180, +180].
    4. Computes magnitude derivatives (|delta_yaw|, |accel_yaw|, etc.).
    5. Computes 3D relative distance from player position to target hitbox.
    """
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accepts either pandas DataFrame or numpy array / dict
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            # Assume 14-column array matching raw feature order
            cols = [
                'yaw', 'pitch', 'delta_yaw', 'delta_pitch',
                'accel_yaw', 'accel_pitch', 'target_x', 'target_y', 'target_z',
                'position_x', 'position_y', 'position_z', 'sensitivity', 'new_sequence'
            ]
            df = pd.DataFrame(X, columns=cols[:X.shape[1]])

        # 1. Clean Sensitivity: fix negative logging artifacts
        df['sensitivity'] = df['sensitivity'].abs().clip(lower=0.1, upper=300.0)

        # 2. Boolean to int
        df['new_sequence'] = df['new_sequence'].astype(int)

        # 3. Wrap unbounded yaw to [-180, 180]
        df['yaw_wrapped'] = ((df['yaw'] + 180.0) % 360.0) - 180.0

        # 4. Magnitude of angular kinematics (critical non-linear signal)
        df['abs_delta_yaw'] = df['delta_yaw'].abs()
        df['abs_delta_pitch'] = df['delta_pitch'].abs()
        df['abs_accel_yaw'] = df['accel_yaw'].abs()
        df['abs_accel_pitch'] = df['accel_pitch'].abs()

        # 5. 3D Euclidean distance to target
        dx = df['target_x'] - df['position_x']
        dy = df['target_y'] - df['position_y']
        dz = df['target_z'] - df['position_z']
        df['distance_to_target'] = np.sqrt(dx**2 + dy**2 + dz**2)

        # Retain model features
        output_features = [
            'delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch',
            'abs_delta_yaw', 'abs_delta_pitch', 'abs_accel_yaw', 'abs_accel_pitch',
            'yaw_wrapped', 'pitch', 'distance_to_target',
            'position_x', 'position_y', 'position_z',
            'target_x', 'target_y', 'target_z',
            'sensitivity', 'new_sequence'
        ]
        self.feature_names_ = output_features
        return df[output_features].values

def build_preprocessing_pipeline():
    """
    Constructs a unified, leakage-safe scikit-learn Pipeline
    reused identically for training and real-time prediction.
    """
    pipeline = Pipeline([
        ('transformer', TelemetryFeatureTransformer()),
        ('scaler', RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0)))
    ])
    return pipeline

def perform_group_aware_split(dataset_path="dataset/final_dataset.csv", group_col='uuid', test_size=0.20, random_state=42):
    """
    Splits dataset using GroupShuffleSplit based on player UUID to eliminate both
    pair-level context leakage and player-identity memorization.
    """
    df = pd.read_csv(dataset_path)

    # Raw feature matrix (excluding direct identifiers and target)
    raw_features = [
        'yaw', 'pitch', 'delta_yaw', 'delta_pitch',
        'accel_yaw', 'accel_pitch', 'target_x', 'target_y', 'target_z',
        'position_x', 'position_y', 'position_z', 'sensitivity', 'new_sequence'
    ]
    X_raw = df[raw_features]
    y = df['cheat'].values
    groups = df[group_col].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X_raw, y, groups))

    X_train_raw = X_raw.iloc[train_idx].copy()
    X_test_raw = X_raw.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Verify zero group overlap
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Critical Data Leakage: {len(overlap)} groups overlap!"

    # Fit pipeline exclusively on training data
    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train_raw)
    X_test_processed = pipeline.transform(X_test_raw)

    # Save fitted pipeline artifact
    os.makedirs("ml/model", exist_ok=True)
    pipeline_path = "ml/model/preprocessing_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    print(f">> Group-Aware Split Completed ({group_col}):")
    print(f"   Train Set: {len(X_train_processed):,} samples across {len(train_groups):,} {group_col}s")
    print(f"   Test Set:  {len(X_test_processed):,} samples across {len(test_groups):,} {group_col}s")
    print(f"   Train Class Balance: Legit={(y_train == 0).sum()} ({y_train.mean():.1%}), Cheat={(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
    print(f"   Test Class Balance:  Legit={(y_test == 0).sum()} ({y_test.mean():.1%}), Cheat={(y_test == 1).sum()} ({(y_test == 1).mean():.1%})")
    print(f"   Group Overlap: {len(overlap)} (Guaranteed 100% Leakage-Free)")
    print(f">> Preprocessing Pipeline persisted to: {pipeline_path}")

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'pipeline': pipeline,
        'train_idx': train_idx,
        'test_idx': test_idx
    }

if __name__ == '__main__':
    perform_group_aware_split()
