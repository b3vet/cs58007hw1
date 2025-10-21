"""
Visualization script to verify step counting
Shows the signal processing steps and detected peaks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stepcount_enhanced import (
    estimate_sampling_rate,
    count_steps_method1_peak_detection,
    count_steps_method2_autocorrelation,
    count_steps_method3_zero_crossings,
    count_steps_combined
)


def visualize_step_detection(csv_path, output_path=None):
    """
    Create comprehensive visualization of step detection process
    """
    # Read data
    df = pd.read_csv(csv_path)
    if 'x' in df.columns:
        df.rename(columns={'x': 'accX', 'y': 'accY', 'z': 'accZ'}, inplace=True)

    # Get timing info
    timestamps = df['seconds_elapsed'].values
    sampling_rate = estimate_sampling_rate(timestamps)
    duration = timestamps[-1] - timestamps[0]

    # Run all methods
    result1 = count_steps_method1_peak_detection(df, sampling_rate)
    result2 = count_steps_method2_autocorrelation(df, sampling_rate)
    result3 = count_steps_method3_zero_crossings(df, sampling_rate)
    combined = count_steps_combined(df, sampling_rate)

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(f'Step Detection Analysis\n{os.path.basename(os.path.dirname(csv_path))}',
                 fontsize=14, fontweight='bold')

    t = timestamps - timestamps[0]  # Time from 0

    # ========== Row 1: Raw data and gravity removal ==========
    ax = axes[0, 0]
    ax.plot(t, df['accX'], label='X', alpha=0.7, linewidth=0.8)
    ax.plot(t, df['accY'], label='Y', alpha=0.7, linewidth=0.8)
    ax.plot(t, df['accZ'], label='Z', alpha=0.7, linewidth=0.8)
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Raw Accelerometer Data')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    lx, ly, lz = result1['linear_acc']
    gx, gy, gz = result1['gravity']
    ax.plot(t, gx, label='Gravity X', linestyle='--', alpha=0.8)
    ax.plot(t, gy, label='Gravity Y', linestyle='--', alpha=0.8)
    ax.plot(t, gz, label='Gravity Z', linestyle='--', alpha=0.8)
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Estimated Gravity (Low-pass filtered)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # ========== Row 2: Method 1 - Peak Detection ==========
    ax = axes[1, 0]
    mag = result1['magnitude']
    mag_smooth = result1['magnitude_smooth']
    peaks = result1['peak_indices']
    threshold = result1['threshold']

    ax.plot(t, mag, alpha=0.4, color='gray', linewidth=0.8, label='Raw magnitude')
    ax.plot(t, mag_smooth, color='blue', linewidth=1.5, label='Smoothed magnitude')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.3f})')
    if len(peaks) > 0:
        ax.scatter(t[peaks], mag_smooth[peaks], color='red', s=50, zorder=5,
                  marker='x', linewidths=2, label=f'Peaks ({len(peaks)})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude (m/s²)')
    ax.set_title(f'Method 1: Peak Detection → {result1["steps"]} steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Histogram of magnitudes
    ax = axes[1, 1]
    ax.hist(mag_smooth, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.axvline(np.mean(mag_smooth), color='green', linestyle=':', linewidth=2, label='Mean')
    ax.set_xlabel('Magnitude (m/s²)')
    ax.set_ylabel('Frequency')
    ax.set_title('Magnitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ========== Row 3: Method 2 - Autocorrelation ==========
    ax = axes[2, 0]
    if 'autocorr' in result2 and len(result2.get('autocorr', [])) > 0:
        autocorr = result2['autocorr']
        lags = np.arange(len(autocorr)) / sampling_rate

        # Plot autocorrelation up to 3 seconds
        max_lag_plot = min(int(3 * sampling_rate), len(autocorr))
        ax.plot(lags[:max_lag_plot], autocorr[:max_lag_plot], color='purple', linewidth=1.5)

        if 'peak_lag' in result2 and result2['peak_lag'] > 0:
            peak_lag = result2['peak_lag']
            peak_time = peak_lag / sampling_rate
            ax.axvline(peak_time, color='red', linestyle='--', linewidth=2,
                      label=f'Period: {peak_time:.3f}s ({1/peak_time:.2f} Hz)')

        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'Method 2: Autocorrelation → {result2["steps"]} steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Autocorrelation failed', ha='center', va='center')
        ax.set_title(f'Method 2: Autocorrelation → {result2["steps"]} steps')

    # Step frequency comparison
    ax = axes[2, 1]
    methods = ['Peak\nDetection', 'Auto-\ncorrelation', 'Zero\nCrossings', 'Combined\n(Median)']
    steps = [
        result1['steps'],
        result2['steps'],
        result3['steps'],
        combined['final_steps']
    ]
    colors = ['blue', 'purple', 'orange', 'green']
    bars = ax.bar(methods, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, step in zip(bars, steps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{step}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Step Count')
    ax.set_title('Method Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(steps) * 1.2)

    # ========== Row 4: Method 3 - Zero Crossings ==========
    ax = axes[3, 0]
    mag_centered = result3['magnitude_centered']
    mag_smooth_zc = result3['magnitude_smooth']

    ax.plot(t, mag_centered, alpha=0.4, color='gray', linewidth=0.8, label='Centered')
    ax.plot(t, mag_smooth_zc, color='orange', linewidth=1.5, label='Smoothed')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    # Mark zero crossings
    signs = np.sign(mag_smooth_zc)
    zero_cross_idx = np.where(np.diff(signs) != 0)[0]
    if len(zero_cross_idx) > 0:
        ax.scatter(t[zero_cross_idx], mag_smooth_zc[zero_cross_idx],
                  color='red', s=20, zorder=5, marker='o', alpha=0.6,
                  label=f'Zero crossings ({len(zero_cross_idx)})')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude (m/s²)')
    ax.set_title(f'Method 3: Zero Crossings → {result3["steps"]} steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = axes[3, 1]
    ax.axis('off')

    summary_text = f"""
SUMMARY STATISTICS
{'='*35}

File: {os.path.basename(os.path.dirname(csv_path))}
Sampling Rate: {sampling_rate:.2f} Hz
Duration: {duration:.2f} seconds
Data Points: {len(df)}

STEP COUNTS:
  Method 1 (Peak Detection):  {result1['steps']} steps
  Method 2 (Autocorrelation): {result2['steps']} steps
  Method 3 (Zero Crossings):  {result3['steps']} steps
  ────────────────────────────────────
  Combined (Median):          {combined['final_steps']} steps

CADENCE:
  {(combined['final_steps'] / duration * 60):.1f} steps/minute

EXPECTED MANUAL COUNT:
  (Count your actual steps for validation)

  For ~21 seconds of walking at normal
  pace: typically 30-45 steps
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()

    return combined


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python visualize_steps.py <path_to_accelerometer.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Generate output filename
    base_name = os.path.basename(os.path.dirname(csv_path))
    output_dir = os.path.join(os.path.dirname(__file__), "../plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_step_analysis.png")

    print(f"\nAnalyzing: {csv_path}")
    result = visualize_step_detection(csv_path, output_path)

    # Calculate cadence
    df_temp = pd.read_csv(csv_path)
    duration = df_temp['seconds_elapsed'].values[-1] - df_temp['seconds_elapsed'].values[0]
    cadence = (result['final_steps'] / duration) * 60

    print(f"\nFinal result: {result['final_steps']} steps")
    print(f"Cadence: {cadence:.1f} steps/minute")
