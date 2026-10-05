"""
Data Preprocessing & Group-Aware Splitting Pipeline
Project: In-Game Cheat Detection System
File: ml/preprocessing.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'yaw', 'pitch', 'delta_yaw', 'delta_pitch', 
    'accel_yaw', 'accel_pitch', 
    'target_x', 'target_y', 'target_z', 
    'position_x', 'position_y', 'position_z', 
    'sensitivity', 'new_sequence'
]
TARGET_COLUMN = 'cheat'

def load_and_preprocess_dataset(dataset_path="dataset/final_dataset.csv"):
    """
    Loads dataset, isolates predictive features, generates group identifiers,
    and strips direct metadata identifiers (uuid, time) to prevent memorization.
    """
    if not os.path.exists(dataset_path):
        fallback = os.path.join(os.path.dirname(__file__), "..", dataset_path)
        if os.path.exists(fallback):
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
    df = pd.read_csv(dataset_path)
    
    # Construct leakage-safe composite group identifier
    groups = df['uuid'].astype(str) + "_" + df['time'].astype(str)
    
    # Feature matrix X and target vector y
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    return X, y, groups, df

def perform_group_aware_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Performs leakage-safe GroupShuffleSplit to guarantee that paired ticks/gameplay
    moments belonging to the same (uuid + time) group do not bleed across train and test sets.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Verify group separation
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    assert len(overlap) == 0, f"Critical Data Leakage Detected: {len(overlap)} groups overlap!"
    
    print(f">> Train Set: {len(X_train)} samples across {len(train_groups)} groups")
    print(f">> Test Set:  {len(X_test)} samples across {len(test_groups)} groups")
    print(f">> Group Overlap: {len(overlap)} (Leakage-Safe Guaranteed)")
    
    return X_train, X_test, y_train, y_test

def build_and_save_scaler(X_train, save_path="ml/model/preprocessing_pipeline.pkl"):
    """
    Fits StandardScaler strictly on training set to avoid data snooping.
    Persists scaler as joblib pipeline.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, save_path)
    print(f">> Scaler saved to: {save_path}")
    return scaler

def get_prepared_data(dataset_path="dataset/final_dataset.csv", scaler_save_path="ml/model/preprocessing_pipeline.pkl"):
    """
    Complete end-to-end preprocessing pipeline returning scaled arrays ready for modeling.
    """
    X, y, groups, _ = load_and_preprocess_dataset(dataset_path)
    X_train, X_test, y_train, y_test = perform_group_aware_split(X, y, groups)
    
    scaler = build_and_save_scaler(X_train, scaler_save_path)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': FEATURE_COLUMNS,
        'scaler': scaler
    }

if __name__ == '__main__':
    print("Testing Preprocessing Pipeline...")
    data = get_prepared_data()
    print(f"X_train_scaled shape: {data['X_train_scaled'].shape}")
    print(f"X_test_scaled shape:  {data['X_test_scaled'].shape}")
