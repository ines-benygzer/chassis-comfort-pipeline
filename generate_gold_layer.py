#!/usr/bin/env python3
"""
Batch Gold Layer Generator
Reads Silver layer Parquet files and generates Gold layer metrics with FFT analysis
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Paths
BASE_PATH = "/Users/inesbenyghzer/.gemini/antigravity/scratch/vehicle_comfort_pipeline"
SILVER_DIR = os.path.join(BASE_PATH, "data/silver/chassis_sensors_clean")
GOLD_DIR = os.path.join(BASE_PATH, "data/gold/chassis_comfort_metrics")

SAMPLING_RATE = 20  # Hz
WINDOW_SIZE = 10  # seconds
SLIDE_SIZE = 2   # seconds

def compute_fft_metrics(acc_z_array):
    """Compute FFT metrics from acceleration array"""
    if len(acc_z_array) < 4:
        return {"dominant_freq": 0.0, "total_energy": 0.0, "band_energy_4_8": 0.0}
    
    y = np.array(acc_z_array, dtype=float)
    n = len(y)
    
    # Real FFT
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(n, 1/SAMPLING_RATE)
    
    # Power spectrum
    powers = np.abs(yf)**2
    
    # Total energy
    total_energy = float(np.sum(powers))
    
    # Dominant frequency (skip DC)
    if len(powers) > 1:
        dom_idx = int(np.argmax(powers[1:])) + 1
        dominant_freq = float(xf[dom_idx])
    else:
        dominant_freq = 0.0
    
    # Band energy 4-8 Hz
    mask = (xf >= 4.0) & (xf <= 8.0)
    band_energy = float(np.sum(powers[mask]))
    
    return {
        "dominant_freq": dominant_freq,
        "total_energy": total_energy,
        "band_energy_4_8": band_energy
    }

def compute_comfort_score(rms, peak, band_energy):
    """Compute physics-based comfort score"""
    # Normalize
    rms_norm = min(rms / 2.0, 1.0)
    peak_norm = min(peak / 3.0, 1.0)
    band_norm = min(band_energy / 50.0, 1.0)
    
    # Formula: 100 - (0.4 * RMS + 0.2 * Peak + 0.4 * Band) * 100
    score = 100.0 - (0.4 * rms_norm + 0.2 * peak_norm + 0.4 * band_norm) * 100.0
    return max(0.0, score)

def process_silver_to_gold():
    """Process Silver layer data to generate Gold layer metrics"""
    print(f"Reading Silver layer from: {SILVER_DIR}")
    
    # Read all Silver data
    df = pd.read_parquet(SILVER_DIR)
    print(f"Loaded {len(df)} records from Silver layer")
    
    # Convert timestamp to datetime
    df['event_time'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('event_time')
    
    # Group by vehicle and test
    results = []
    
    for (vehicle_id, test_id), group in df.groupby(['vehicle_id', 'test_id']):
        print(f"Processing {vehicle_id} - {test_id}: {len(group)} records")
        
        # Create time windows
        group = group.set_index('event_time')
        
        # Sliding windows
        window_start = group.index.min()
        window_end = group.index.max()
        
        current_time = window_start
        while current_time < window_end:
            end_time = current_time + pd.Timedelta(seconds=WINDOW_SIZE)
            
            # Get data in window
            window_data = group[(group.index >= current_time) & (group.index < end_time)]
            
            if len(window_data) >= 4:  # Minimum data points
                acc_z = window_data['acc_z'].values
                
                # Basic metrics
                rms_acc_z = np.sqrt(np.mean(acc_z**2))
                peak_acc_z = np.max(np.abs(acc_z))
                avg_speed = window_data['speed_kmh'].mean()
                
                # FFT metrics
                fft_metrics = compute_fft_metrics(acc_z)
                
                # Comfort score
                comfort_score = compute_comfort_score(
                    rms_acc_z, 
                    peak_acc_z, 
                    fft_metrics['band_energy_4_8']
                )
                
                results.append({
                    'start_time': current_time,
                    'end_time': end_time,
                    'vehicle_id': vehicle_id,
                    'test_id': test_id,
                    'rms_acc_z': rms_acc_z,
                    'peak_acc_z': peak_acc_z,
                    'avg_speed': avg_speed,
                    'dominant_freq': fft_metrics['dominant_freq'],
                    'total_energy': fft_metrics['total_energy'],
                    'band_energy_4_8': fft_metrics['band_energy_4_8'],
                    'comfort_score': comfort_score
                })
            
            # Slide window
            current_time += pd.Timedelta(seconds=SLIDE_SIZE)
    
    # Create Gold DataFrame
    gold_df = pd.DataFrame(results)
    
    if len(gold_df) > 0:
        # Add partition columns
        gold_df['year'] = gold_df['start_time'].dt.year
        gold_df['month'] = gold_df['start_time'].dt.month
        gold_df['day'] = gold_df['start_time'].dt.day
        
        # Write to Parquet with partitioning
        print(f"\nWriting {len(gold_df)} Gold layer records to: {GOLD_DIR}")
        gold_df.to_parquet(
            GOLD_DIR,
            partition_cols=['year', 'month', 'day'],
            engine='pyarrow',
            index=False
        )
        print("✓ Gold layer generated successfully!")
        
        # Display sample
        print("\nSample Gold Layer Metrics:")
        print(gold_df[['vehicle_id', 'rms_acc_z', 'peak_acc_z', 'dominant_freq', 'comfort_score']].head(10))
    else:
        print("No data to process")

if __name__ == "__main__":
    process_silver_to_gold()
