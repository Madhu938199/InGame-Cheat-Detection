"""
Stage 2: Exploratory Data Analysis (EDA) Plotting Script
Generates clean, informative visualizations from dataset/final_dataset.csv
Saves figures into ml/plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_visualizations(dataset_path="dataset/final_dataset.csv", output_dir="ml/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    
    # Styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    sns.set_palette(['#10b981', '#ef4444']) # Green: Legit (0), Red: Cheater (1)
    
    # -------------------------------------------------------------
    # Plot 1: Target Class Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['cheat'].value_counts()
    bars = ax.bar(['Legitimate (0)', 'Suspicious / Cheater (1)'], counts.values, color=['#10b981', '#ef4444'], width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{yval:,} ({yval/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 6000)
    ax.set_ylabel('Number of Telemetry Records', fontsize=11)
    ax.set_title('Target Class Balance (cheat)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_class_distribution.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: Kinematic Histograms (Clipped / Log Views)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    movement_features = ['delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch']
    clip_ranges = {
        'delta_yaw': (-60, 60),
        'delta_pitch': (-25, 25),
        'accel_yaw': (-50, 50),
        'accel_pitch': (-30, 30)
    }
    
    for ax, feat in zip(axes.flatten(), movement_features):
        c_min, c_max = clip_ranges[feat]
        legit = df[df['cheat'] == 0][feat].clip(c_min, c_max)
        cheat = df[df['cheat'] == 1][feat].clip(c_min, c_max)
        
        sns.histplot(legit, ax=ax, color='#10b981', label='Legitimate (0)', kde=True, stat="density", bins=40, alpha=0.4)
        sns.histplot(cheat, ax=ax, color='#ef4444', label='Cheater (1)', kde=True, stat="density", bins=40, alpha=0.4)
        ax.set_title(f'Distribution of {feat} (Clipped [{c_min}, {c_max}])', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{feat} (degrees/tick or degrees/tick²)')
        ax.set_ylabel('Density')
        ax.legend()
        
    plt.suptitle('Kinematic Movement Distributions: Legitimate vs. Cheater', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_kinematic_histograms.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Box Plots Comparing Variances & Heavy Tails
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, feat in zip(axes, movement_features):
        c_min, c_max = clip_ranges[feat]
        df_clipped = df.copy()
        df_clipped[feat] = df_clipped[feat].clip(c_min, c_max)
        sns.boxplot(x='cheat', y=feat, data=df_clipped, ax=ax, palette=['#10b981', '#ef4444'], width=0.4)
        ax.set_xticklabels(['Legitimate (0)', 'Cheater (1)'])
        ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Clipped Value')
        
    plt.suptitle('Box Plots of Angular Velocity & Acceleration (IQR and Tail Spread)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '3_kinematic_boxplots.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 4: Correlation Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_cols = [
        'delta_yaw', 'delta_pitch', 'accel_yaw', 'accel_pitch',
        'yaw', 'pitch', 'target_x', 'target_y', 'target_z',
        'position_x', 'position_y', 'position_z', 'sensitivity', 'cheat'
    ]
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'}, ax=ax, square=True)
    ax.set_title('Feature Correlation Matrix (Including Target "cheat")', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 5: Cheater vs Legitimate Scatter / Phase Space Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    sub_df = df.sample(n=2000, random_state=42)
    sns.scatterplot(
        data=sub_df, 
        x=sub_df['delta_yaw'].clip(-100, 100), 
        y=sub_df['accel_yaw'].clip(-150, 150), 
        hue='cheat', 
        palette=['#10b981', '#ef4444'], 
        alpha=0.6, 
        s=35,
        ax=ax
    )
    ax.set_title('Phase Space: Delta Yaw vs. Accel Yaw (Subsampled 2,000 points)', fontsize=13, fontweight='bold')
    ax.set_xlabel('delta_yaw (Clipped [-100, 100])')
    ax.set_ylabel('accel_yaw (Clipped [-150, 150])')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Legitimate (0)', 'Cheater (1)'])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_phase_space_scatter.png'), dpi=300)
    plt.close()
    
    print(f"All 5 EDA visualizations saved successfully to: {output_dir}")

if __name__ == '__main__':
    generate_eda_visualizations()
