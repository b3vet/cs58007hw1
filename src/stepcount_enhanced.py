"""
Enhanced Step Counter with Multiple Methods
Uses only numpy and pandas for processing
Implements: magnitude, peak detection, zero-crossings, low-pass filtering, convolution
"""

import numpy as np
import pandas as pd

# ==================== SIGNAL PROCESSING UTILITIES ====================

def estimate_sampling_rate(timestamps):
    """Estimate sampling rate from timestamps"""
    dt = np.diff(timestamps)
    dt = dt[dt > 0]  # Remove zeros
    median_dt = np.median(dt)
    return 1.0 / median_dt if median_dt > 0 else 100.0


def lowpass_filter_exponential(signal, cutoff_hz, sampling_rate):
    """
    Simple first-order IIR low-pass filter (exponential smoothing)
    Formula: y[n] = alpha * y[n-1] + (1 - alpha) * x[n]
    where alpha = exp(-2*pi*fc/fs)
    """
    if cutoff_hz <= 0 or sampling_rate <= 0:
        return signal.copy()

    alpha = np.exp(-2.0 * np.pi * cutoff_hz / sampling_rate)
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]

    for i in range(1, len(signal)):
        filtered[i] = alpha * filtered[i-1] + (1 - alpha) * signal[i]

    return filtered


def convolve_1d(signal, kernel, mode='same'):
    """
    1D convolution implementation   
    Args:
        signal: Input signal array
        kernel: Convolution kernel
        mode: 'same' to return output of same size as signal
    
    Returns:
        Convolved signal
    """
    n = len(signal)
    k = len(kernel)
    
    if mode == 'same':
        # Pad signal to keep same size
        pad = k // 2
        padded = np.pad(signal, (pad, pad), mode='edge')
        result = np.zeros(n)
        
        for i in range(n):
            result[i] = np.sum(padded[i:i+k] * kernel)
        
        return result
    else:
        # Full convolution
        result = np.zeros(n + k - 1)
        for i in range(len(result)):
            start_k = max(0, k - 1 - i)
            end_k = min(k, n + k - 1 - i)
            start_s = max(0, i - k + 1)
            
            for j in range(start_k, end_k):
                result[i] += signal[start_s + j - start_k] * kernel[j]
        
        return result


def lowpass_filter_moving_average(signal, window_size):
    """
    Simple moving average low-pass filter (convolution with box kernel)
    This is convolution with a rectangular window
    """
    if window_size <= 1:
        return signal.copy()

    # Create box kernel (all ones, normalized)
    kernel = np.ones(window_size) / window_size

    # Convolve signal with kernel (mode='same' keeps same length)
    # Using manual convolution to avoid scipy dependency
    filtered = convolve_1d(signal, kernel, mode='same')

    return filtered


def compute_magnitude(x, y, z):
    """Compute magnitude of 3D acceleration vector"""
    return np.sqrt(x**2 + y**2 + z**2)


def remove_gravity(ax, ay, az, cutoff_hz, sampling_rate):
    """
    Remove gravity component using low-pass filter
    Gravity is the low-frequency component
    Linear acceleration is high-frequency component
    """
    # Estimate gravity as low-pass filtered signal
    gx = lowpass_filter_exponential(ax, cutoff_hz, sampling_rate)
    gy = lowpass_filter_exponential(ay, cutoff_hz, sampling_rate)
    gz = lowpass_filter_exponential(az, cutoff_hz, sampling_rate)

    # Linear acceleration = total - gravity
    lx = ax - gx
    ly = ay - gy
    lz = az - gz

    return lx, ly, lz, gx, gy, gz


def detect_peaks(signal, min_height=None, min_distance=1):
    """
    Detect peaks in signal
    Peak: point that is higher than both neighbors

    Args:
        signal: 1D array
        min_height: minimum height threshold
        min_distance: minimum distance between peaks (in samples)

    Returns:
        Array of peak indices
    """
    peaks = []

    # Find local maxima
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if min_height is None or signal[i] >= min_height:
                peaks.append(i)

    # Enforce minimum distance
    if min_distance > 1 and len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
            else:
                # Keep the higher peak
                if signal[peak] > signal[filtered_peaks[-1]]:
                    filtered_peaks[-1] = peak
        peaks = filtered_peaks

    return np.array(peaks)


def count_zero_crossings(signal):
    """
    Count zero crossings in signal
    A zero crossing occurs when signal changes sign
    """
    # Find where sign changes
    signs = np.sign(signal)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    return len(zero_crossings)


def detect_steps_from_zero_crossings(signal, sampling_rate, expected_step_freq_range=(0.5, 3.0)):
    """
    Alternative method: estimate steps from zero-crossing rate
    Walking is periodic, so zero-crossings relate to step frequency
    """
    zcr = count_zero_crossings(signal)
    duration = len(signal) / sampling_rate

    # Zero crossings per second
    zcr_per_sec = zcr / duration

    # Each step cycle creates approximately 2 zero crossings (up and down)
    # So step frequency ≈ ZCR / 2
    estimated_step_freq = zcr_per_sec / 2.0

    # Clamp to reasonable range
    min_freq, max_freq = expected_step_freq_range
    if estimated_step_freq < min_freq:
        estimated_step_freq = min_freq
    elif estimated_step_freq > max_freq:
        estimated_step_freq = max_freq

    # Total steps = frequency × duration
    steps = int(estimated_step_freq * duration)

    return steps, estimated_step_freq, zcr


# ==================== MAIN STEP COUNTING METHODS ====================

def count_steps_method1_peak_detection(df, sampling_rate):
    """
    Method 1: Peak Detection on Magnitude
    - Compute magnitude of linear acceleration
    - Remove gravity using low-pass filter
    - Smooth the signal
    - Detect peaks
    """
    # Extract axes
    ax = df['accX'].values
    ay = df['accY'].values
    az = df['accZ'].values

    # Remove gravity (low-pass at 0.3 Hz)
    lx, ly, lz, gx, gy, gz = remove_gravity(ax, ay, az, cutoff_hz=0.3, sampling_rate=sampling_rate)

    # Compute magnitude of linear acceleration
    mag = compute_magnitude(lx, ly, lz)

    # Smooth using moving average (convolution)
    window_size = max(3, int(sampling_rate * 0.05))  # 50ms window
    mag_smooth = lowpass_filter_moving_average(mag, window_size)

    # Adaptive threshold: mean + fraction of std
    threshold = np.mean(mag_smooth) + 0.3 * np.std(mag_smooth)

    # Minimum distance between steps (based on typical step frequency)
    # Typical walking: 1-2 steps/second, so min distance = 0.3s
    min_distance_samples = int(sampling_rate * 0.3)

    # Detect peaks
    peaks = detect_peaks(mag_smooth, min_height=threshold, min_distance=min_distance_samples)

    return {
        'steps': len(peaks),
        'peak_indices': peaks,
        'magnitude': mag,
        'magnitude_smooth': mag_smooth,
        'threshold': threshold,
        'linear_acc': (lx, ly, lz),
        'gravity': (gx, gy, gz)
    }


def count_steps_method2_autocorrelation(df, sampling_rate):
    """
    Method 2: Autocorrelation to find periodicity
    - Compute magnitude
    - Use autocorrelation to find dominant period
    - Estimate steps from period
    """
    ax = df['accX'].values
    ay = df['accY'].values
    az = df['accZ'].values

    # Remove gravity
    lx, ly, lz, _, _, _ = remove_gravity(ax, ay, az, cutoff_hz=0.3, sampling_rate=sampling_rate)
    mag = compute_magnitude(lx, ly, lz)

    # Normalize signal (zero mean, unit variance)
    mag_normalized = (mag - np.mean(mag)) / (np.std(mag) + 1e-10)

    # Compute autocorrelation manually (convolution of signal with itself)
    # We only need lags from 0.3s to 2s (reasonable step period range)
    min_lag = int(0.3 * sampling_rate)  # Min 0.3s between steps
    max_lag = int(2.0 * sampling_rate)  # Max 2s between steps

    autocorr = np.correlate(mag_normalized, mag_normalized, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags

    # Find peak in autocorrelation (ignore lag=0)
    if max_lag < len(autocorr):
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) > 0:
            peak_lag = min_lag + np.argmax(search_region)
            step_period = peak_lag / sampling_rate
            step_frequency = 1.0 / step_period

            duration = len(mag) / sampling_rate
            steps = int(step_frequency * duration)

            return {
                'steps': steps,
                'step_frequency': step_frequency,
                'step_period': step_period,
                'autocorr': autocorr,
                'peak_lag': peak_lag
            }

    # Fallback
    return {'steps': 0, 'step_frequency': 0, 'step_period': 0}


def count_steps_method3_zero_crossings(df, sampling_rate):
    """
    Method 3: Zero Crossing Rate
    - Compute magnitude
    - High-pass filter to remove DC offset
    - Count zero crossings
    - Estimate steps from crossing rate
    """
    ax = df['accX'].values
    ay = df['accY'].values
    az = df['accZ'].values

    # Remove gravity
    lx, ly, lz, _, _, _ = remove_gravity(ax, ay, az, cutoff_hz=0.3, sampling_rate=sampling_rate)
    mag = compute_magnitude(lx, ly, lz)

    # Remove DC component (subtract mean)
    mag_centered = mag - np.mean(mag)

    # Smooth to reduce noise-induced zero crossings
    window_size = max(3, int(sampling_rate * 0.05))
    mag_smooth = lowpass_filter_moving_average(mag_centered, window_size)

    # Count steps from zero crossings
    steps, step_freq, zcr = detect_steps_from_zero_crossings(
        mag_smooth,
        sampling_rate,
        expected_step_freq_range=(0.8, 2.5)
    )

    return {
        'steps': steps,
        'step_frequency': step_freq,
        'zero_crossings': zcr,
        'magnitude_centered': mag_centered,
        'magnitude_smooth': mag_smooth
    }


def count_steps_combined(df, sampling_rate):
    """
    Combined approach: Use multiple methods and take consensus
    """
    result1 = count_steps_method1_peak_detection(df, sampling_rate)
    result2 = count_steps_method2_autocorrelation(df, sampling_rate)
    result3 = count_steps_method3_zero_crossings(df, sampling_rate)

    steps = [result1['steps'], result2['steps'], result3['steps']]

    # Use median as robust estimate
    final_steps = int(np.median(steps))

    return {
        'final_steps': final_steps,
        'method1_steps': result1['steps'],
        'method2_steps': result2['steps'],
        'method3_steps': result3['steps'],
        'result1': result1,
        'result2': result2,
        'result3': result3
    }


# ==================== MAIN INTERFACE ====================

def count_steps_from_csv(csv_path, method='peak_detection', verbose=True):
    """
    Main function to count steps from accelerometer CSV file

    Args:
        csv_path: Path to Accelerometer.csv file
        method: 'peak_detection', 'autocorrelation', 'zero_crossings', or 'combined'
        verbose: Print detailed results

    Returns:
        Dictionary with step count and additional information
    """
    # Read data
    df = pd.read_csv(csv_path)

    # Rename columns to standard format
    if 'x' in df.columns:
        df.rename(columns={'x': 'accX', 'y': 'accY', 'z': 'accZ'}, inplace=True)

    # Estimate sampling rate
    timestamps = df['seconds_elapsed'].values
    sampling_rate = estimate_sampling_rate(timestamps)
    duration = timestamps[-1] - timestamps[0]

    # Choose method
    if method == 'peak_detection':
        result = count_steps_method1_peak_detection(df, sampling_rate)
        steps = result['steps']
    elif method == 'autocorrelation':
        result = count_steps_method2_autocorrelation(df, sampling_rate)
        steps = result['steps']
    elif method == 'zero_crossings':
        result = count_steps_method3_zero_crossings(df, sampling_rate)
        steps = result['steps']
    elif method == 'combined':
        result = count_steps_combined(df, sampling_rate)
        steps = result['final_steps']
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate cadence (steps per minute)
    cadence = (steps / duration) * 60.0 if duration > 0 else 0

    # Prepare output
    output = {
        'csv_path': csv_path,
        'sampling_rate': sampling_rate,
        'duration_sec': duration,
        'steps': steps,
        'cadence_spm': cadence,
        'method': method,
        'details': result
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step Counter - Method: {method}")
        print(f"{'='*60}")
        print(f"File: {csv_path}")
        print(f"Sampling rate: {sampling_rate:.2f} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Estimated steps: {steps}")
        print(f"Cadence: {cadence:.1f} steps/minute")

        if method == 'combined':
            print(f"\nMethod breakdown:")
            print(f"  - Peak detection:   {result['method1_steps']} steps")
            print(f"  - Autocorrelation:  {result['method2_steps']} steps")
            print(f"  - Zero crossings:   {result['method3_steps']} steps")
            print(f"  - Final (median):   {steps} steps")

        print(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stepcount_enhanced.py <path_to_accelerometer.csv> [method]")
        print("Methods: peak_detection (default), autocorrelation, zero_crossings, combined")
        sys.exit(1)

    csv_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'peak_detection'

    result = count_steps_from_csv(csv_path, method=method, verbose=True)
