
import os
import pandas as pd
import numpy as np
from deltalake import DeltaTable
import time

GOLD_DIR = "/Users/inesbenyghzer/.gemini/antigravity/scratch/vehicle_comfort_pipeline/data/gold/chassis_comfort_delta"
SILVER_DIR = "/Users/inesbenyghzer/.gemini/antigravity/scratch/vehicle_comfort_pipeline/data/silver/chassis_sensors_delta"

def measure_throughput():
    print("--- Reliability Measurement (Recent) ---")
    
    # 1. Check Gold Table Data
    if os.path.exists(GOLD_DIR):
        dt = DeltaTable(GOLD_DIR)
        df = dt.to_pandas()
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Filter for new scenarios specifically
        scenarios = ['Run_Pothole_Alley', 'Run_Urban_Road', 'Run_Smooth_Highway']
        recent_df = df[df['test_id'].str.contains('|'.join(scenarios))]
        
        print(f"Total Records in Gold: {len(df)}")
        print(f"Recent Records (New Scenarios): {len(recent_df)}")
        
        # Summary by Scenario (New only)
        for scenario in scenarios:
            sub = recent_df[recent_df['test_id'].str.contains(scenario)]
            if not sub.empty:
                avg_w = sub['weighted_acc_z'].mean()
                avg_c = sub['comfort_score'].mean()
                print(f"Scenario {scenario}: Avg ISO Acc={avg_w:.4f}, Avg Comfort={avg_c:.1f}%")

    # 2. Real-time Throughput (Last 5 mins of Silver)
    if os.path.exists(SILVER_DIR):
        dt_s = DeltaTable(SILVER_DIR)
        df_s = dt_s.to_pandas()
        df_s['ingestion_time'] = pd.to_datetime(df_s['ingestion_time'])
        
        now = df_s['ingestion_time'].max()
        last_5m = df_s[df_s['ingestion_time'] > (now - pd.Timedelta(minutes=5))]
        
        print(f"Records in last 5 minutes (Silver): {len(last_5m)}")
        if len(last_5m) > 0:
            duration = (last_5m['ingestion_time'].max() - last_5m['ingestion_time'].min()).total_seconds()
            if duration > 0:
                print(f"Current Throughput: {len(last_5m)/duration:.2f} msg/sec")
            
            # Message integrity: 3 vehicles * 20Hz = 60 msg/sec expected
            print(f"Expected Throughput (3 vehicles @ 20Hz): 60.00 msg/sec")

if __name__ == "__main__":
    measure_throughput()
