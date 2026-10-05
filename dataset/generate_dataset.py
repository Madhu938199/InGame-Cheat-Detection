"""
Dataset Generator for In-Game Cheat Detection System
Generates the 10,000-row telemetry dataset (final_dataset.csv)
matching the exact statistical properties and class distributions:
- 7,528 Legitimate sessions (75.28%)
- 2,472 Suspicious/Cheater sessions (24.72%)
- Realistic paired gameplay groups (uuid + time)
"""

import os
import uuid
import numpy as np
import pandas as pd

def generate_telemetry_dataset(output_path="dataset/final_dataset.csv", num_records=10000, random_seed=42):
    np.random.seed(random_seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_cheaters = 2472
    num_legit = 7528
    
    # Generate unique player UUIDs
    player_pool = [str(uuid.uuid4())[:8] for _ in range(250)]
    
    # We will generate groups so that uuid + time groups naturally exist
    records = []
    
    # Generate 5000 distinct gameplay moments/ticks across players
    # Many moments have 2 ticks (paired records) to emulate paired anti-cheat snapshot logging
    num_groups = 5200
    group_defs = []
    for g_id in range(num_groups):
        p_uuid = np.random.choice(player_pool)
        t_stamp = np.random.randint(1000, 99999)
        # determine whether this group has 1 or 2 records
        size = 2 if g_id < 4800 else 1
        group_defs.append((p_uuid, t_stamp, size))
    
    legit_count = 0
    cheat_count = 0
    
    for p_uuid, t_stamp, size in group_defs:
        if len(records) >= num_records:
            break
            
        # Target for this group: cheater or legit
        # Need to maintain 24.72% cheaters overall
        remaining_needed = num_records - len(records)
        cheaters_remaining = num_cheaters - cheat_count
        prob_cheat = max(0.0, min(1.0, cheaters_remaining / max(1, remaining_needed)))
        
        is_cheat_group = 1 if np.random.rand() < prob_cheat else 0
        
        for _ in range(size):
            if len(records) >= num_records:
                break
                
            label = is_cheat_group
            if label == 1 and cheat_count >= num_cheaters:
                label = 0
            elif label == 0 and legit_count >= num_legit:
                label = 1
                
            if label == 1:
                cheat_count += 1
                # Cheater telemetry: sharp angular snaps, extreme accelerations, robotic locking
                yaw = np.random.uniform(-180.0, 180.0)
                pitch = np.random.uniform(-45.0, 45.0)
                delta_yaw = np.random.normal(28.5, 32.0)
                delta_pitch = np.random.normal(16.2, 19.5)
                accel_yaw = np.random.normal(1.8, 6.2)
                accel_pitch = np.random.normal(1.2, 4.1)
                sensitivity = float(np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], p=[0.1, 0.2, 0.4, 0.2, 0.1]))
            else:
                legit_count += 1
                # Legitimate player telemetry: smooth, human biological tracking, continuous curves
                yaw = np.random.uniform(-180.0, 180.0)
                pitch = np.random.uniform(-80.0, 80.0)
                delta_yaw = np.random.normal(5.2, 18.0)
                delta_pitch = np.random.normal(4.6, 12.0)
                accel_yaw = np.random.normal(-0.6, 4.2)
                accel_pitch = np.random.normal(-0.3, 2.4)
                sensitivity = float(np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], p=[0.15, 0.25, 0.35, 0.15, 0.1]))
                
            target_x = np.round(np.random.uniform(-50.0, 50.0), 3)
            target_y = np.round(np.random.uniform(0.0, 30.0), 3)
            target_z = np.round(np.random.uniform(-50.0, 50.0), 3)
            
            position_x = np.round(np.random.uniform(-100.0, 100.0), 3)
            position_y = np.round(np.random.uniform(0.0, 20.0), 3)
            position_z = np.round(np.random.uniform(-100.0, 100.0), 3)
            
            new_sequence = int(np.random.choice([0, 1], p=[0.85, 0.15]))
            
            records.append({
                'uuid': p_uuid,
                'yaw': round(yaw, 2),
                'pitch': round(pitch, 2),
                'delta_yaw': round(delta_yaw, 2),
                'delta_pitch': round(delta_pitch, 2),
                'accel_yaw': round(accel_yaw, 2),
                'accel_pitch': round(accel_pitch, 2),
                'target_x': target_x,
                'target_y': target_y,
                'target_z': target_z,
                'position_x': position_x,
                'position_y': position_y,
                'position_z': position_z,
                'sensitivity': sensitivity,
                'time': t_stamp,
                'new_sequence': new_sequence,
                'cheat': label
            })
            
    df = pd.DataFrame(records)
    # Clip numerical outliers to match project statistics boundary
    df['delta_yaw'] = df['delta_yaw'].clip(-66.98, 115.58)
    df['delta_pitch'] = df['delta_pitch'].clip(-42.95, 72.02)
    df['accel_yaw'] = df['accel_yaw'].clip(-17.11, 18.73)
    df['accel_pitch'] = df['accel_pitch'].clip(-12.47, 10.49)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records at: {output_path}")
    print(f"Legitimate (0): {(df['cheat'] == 0).sum()} ({(df['cheat'] == 0).mean()*100:.2f}%)")
    print(f"Cheater (1): {(df['cheat'] == 1).sum()} ({(df['cheat'] == 1).mean()*100:.2f}%)")
    return df

if __name__ == '__main__':
    generate_telemetry_dataset()
