import numpy as np

def calculate_iso_weighted_acc(acc_z_list, sampling_rate=20):
    """
    Computes ISO 2631-1 frequency-weighted acceleration (Wk curve).
    This is a copy of the logic in spark_pipeline.py for testing.
    """
    try:
        if acc_z_list is None or len(acc_z_list) < 2:
            return 0.0
        
        y = np.array(acc_z_list)
        n = len(y)
        
        # Remove DC component (detrend)
        y = y - np.mean(y)
        
        # FFT (rfft returns magnitudes for positive frequencies)
        # Normalize FFT by n to get amplitude-like values for RMS calculation
        yf = np.abs(np.fft.rfft(y)) / (n / 2.0)
        xf = np.fft.rfftfreq(n, 1/sampling_rate)
        
        # Wk Weights (ISO 2631-1 Vertical)
        freqs = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0])
        gains = np.array([0.062, 0.176, 0.643, 0.967, 1.000, 0.977, 0.892, 0.776, 0.648, 0.512, 0.409, 0.330, 0.266, 0.215, 0.176, 0.145, 0.119])
        
        # Interpolate gains for each FFT bin
        wk_gains = np.interp(xf, freqs, gains, left=0.0, right=0.0)
        
        # Apply weighting
        weighted_magnitudes = yf * wk_gains
        
        # RMS calculation
        weighted_rms = float(np.sqrt(np.mean(weighted_magnitudes**2)))
        
        return weighted_rms
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

def test_udf():
    fs = 20
    duration = 10 # seconds
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    
    # Test 1: 5 Hz sine wave, amplitude 1.0
    # Expected: Wk gain at 5 Hz is 1.0. 
    # Amplitude 1.0 sine wave has RMS = 1/sqrt(2) approx 0.707.
    # However, the user's formula for weighted_rms is sqrt(mean(weighted_magnitudes**2)).
    # If a_w is computed on amplitudes normalized by n/2, the result for a pure sine should be related to the gain.
    sig_5hz = 1.0 * np.sin(2 * np.pi * 5 * t)
    res_5hz = calculate_iso_weighted_acc(sig_5hz, fs)
    print(f"Test 5Hz (Amp=1.0): Result={res_5hz:.4f} (Expected near 1.0 or gain-related)")

    # Test 2: 1 Hz sine wave, amplitude 1.0
    # Expected: Wk gain at 1 Hz is 0.176.
    sig_1hz = 1.0 * np.sin(2 * np.pi * 1 * t)
    res_1hz = calculate_iso_weighted_acc(sig_1hz, fs)
    print(f"Test 1Hz (Amp=1.0): Result={res_1hz:.4f} (Expected near 0.176 or gain-related)")
    
    print(f"Ratio 1Hz/5Hz: {res_1hz/res_5hz:.4f} (Ideal ISO Ratio: 0.176/1.0 = 0.176)")

if __name__ == "__main__":
    test_udf()
