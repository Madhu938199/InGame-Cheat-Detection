"""
Stage 1 Rigorous Analysis Script
Executes full verification on the real dataset/final_dataset.csv
"""

import pandas as pd
import numpy as np

df = pd.read_csv("dataset/final_dataset.csv")

print("=== 1. SHAPE, DTYPES, MISSING, DUPLICATES ===")
print("Shape:", df.shape)
print("Missing values total:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nDtypes and unique counts:")
for col in df.columns:
    print(f"  {col:<14}: dtype={str(df[col].dtype):<8}, unique={df[col].nunique():<6}")

print("\n=== 2. CLASS BALANCE ===")
print(df['cheat'].value_counts())
print(df['cheat'].value_counts(normalize=True) * 100)

print("\n=== 3. PAIRING ANALYSIS (uuid + time) ===")
df['group_uuid_time'] = df['uuid'].astype(str) + "_" + df['time'].astype(str)
group_counts = df['group_uuid_time'].value_counts()
print(f"Total groups (uuid + time): {len(group_counts)}")
print(f"Min rows in group: {group_counts.min()}, Max rows in group: {group_counts.max()}")

# Check label balance per pair
pair_label_diff = df.groupby('group_uuid_time')['cheat'].agg(['sum', 'count'])
perfect_pairs = ((pair_label_diff['count'] == 2) & (pair_label_diff['sum'] == 1)).sum()
print(f"Groups of exactly 2 with one cheat=0 and one cheat=1: {perfect_pairs} / {len(group_counts)}")

# Check feature identicality within pairs
identical_pcts = {}
cols_to_check = [c for c in df.columns if c not in ['uuid', 'time', 'cheat', 'group_uuid_time']]
for c in cols_to_check:
    diff_count = 0
    # Group by group_uuid_time and check if nunique == 1
    same_count = (df.groupby('group_uuid_time')[c].nunique() == 1).sum()
    pct = (same_count / len(group_counts)) * 100
    identical_pcts[c] = pct
    print(f"  {c:<14}: identical in {pct:.1f}% of pairs")

print("\n=== 4. UUID VS UUID+TIME GROUPING ===")
print(f"Unique players (uuid): {df['uuid'].nunique()}")
print(f"Average groups (uuid+time) per player: {len(group_counts) / df['uuid'].nunique():.2f}")
uuid_group_counts = df['uuid'].value_counts()
print(f"Min rows per uuid: {uuid_group_counts.min()}, Max rows per uuid: {uuid_group_counts.max()}")

print("\n=== 5. SENSITIVITY INVESTIGATION (NEGATIVE VALUES) ===")
neg_sens = df[df['sensitivity'] < 0]
print(f"Rows with sensitivity < 0: {len(neg_sens)}")
print(neg_sens[['uuid', 'time', 'sensitivity', 'cheat', 'yaw', 'delta_yaw']])

print("\n=== 6. SUMMARY STATISTICS & OUTLIERS ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
stats = df[numeric_cols].describe().T[['min', 'mean', 'std', '25%', '50%', '75%', 'max']]
print(stats.to_string())

print("\n=== 7. CORRELATIONS WITH TARGET ('cheat') ===")
corrs = df[numeric_cols].corr()['cheat'].sort_values(ascending=False)
for col, val in corrs.items():
    if col != 'cheat':
        print(f"  {col:<14}: {val:+.4f}")

print("\n=== 8. STD OF MOVEMENT FEATURES BY CLASS ===")
movement_cols = ['delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch']
print(df.groupby('cheat')[movement_cols].std())
