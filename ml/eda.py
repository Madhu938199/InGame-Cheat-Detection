"""
Exploratory Data Analysis (EDA) Module
Project: In-Game Cheat Detection System
File: ml/eda.py
"""

import os
import pandas as pd
import numpy as np

def run_eda(dataset_path="dataset/final_dataset.csv"):
    if not os.path.exists(dataset_path):
        # Check relative fallback
        fallback = os.path.join(os.path.dirname(__file__), "..", dataset_path)
        if os.path.exists(fallback):
            dataset_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}. Please generate or provide final_dataset.csv.")
            
    print("=" * 70)
    print(" STEP 1: EXPLORATORY DATA ANALYSIS & TELEMETRY HEALTH AUDIT")
    print("=" * 70)
    
    df = pd.read_csv(dataset_path)
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n[1] Data Types & Non-Null Values:")
    print("-" * 50)
    print(df.info())
    
    print("\n[2] Missing Values & Duplicate Check:")
    print("-" * 50)
    missing = df.isnull().sum()
    print(f"Total Missing Values: {missing.sum()}")
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    
    print("\n[3] Target Variable Distribution ('cheat'):")
    print("-" * 50)
    counts = df['cheat'].value_counts()
    percentages = df['cheat'].value_counts(normalize=True) * 100
    for label, count in counts.items():
        name = "Legitimate (0)" if label == 0 else "Suspicious/Cheater (1)"
        print(f"  {name}: {count:,} records ({percentages[label]:.2f}%)")
        
    print("\n[4] Telemetry Feature Summary Statistics:")
    print("-" * 70)
    summary_cols = ['delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch', 'sensitivity']
    stats_df = df[summary_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(stats_df.round(2).to_string())
    
    print("\n[5] Group Leakage Analysis (uuid + time pairing):")
    print("-" * 70)
    df['group_id'] = df['uuid'].astype(str) + "_" + df['time'].astype(str)
    unique_groups = df['group_id'].nunique()
    print(f"Total Records: {len(df)}")
    print(f"Unique Groups (uuid + time): {unique_groups}")
    print(f"Average Records per Group: {len(df) / unique_groups:.2f}")
    paired_groups = (df['group_id'].value_counts() > 1).sum()
    print(f"Groups with Multiple Paired Telemetry Ticks: {paired_groups}")
    print(">> Audit Result: Group-Aware splitting (GroupShuffleSplit) is strictly mandatory to avoid leakage!")
    
    print("\n[6] Feature Correlation with Target ('cheat'):")
    print("-" * 70)
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()['cheat'].sort_values(ascending=False)
    for col, c_val in corr.items():
        if col != 'cheat':
            print(f"  {col:<18} : {c_val:+.4f}")
            
    print("=" * 70)
    print(" EDA COMPLETED SUCCESSFULLY.")
    print("=" * 70)
    return df

if __name__ == '__main__':
    run_eda()
