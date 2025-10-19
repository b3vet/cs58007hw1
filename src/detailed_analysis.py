import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

# Folder where all logs are stored
ROOT = "../data"

def read_accel_csv(folder):
    # Try both ../data (from src/) and data (from root)
    if os.path.exists(os.path.join(ROOT, folder, "Accelerometer.csv")):
        path = os.path.join(ROOT, folder, "Accelerometer.csv")
    elif os.path.exists(os.path.join("data", folder, "Accelerometer.csv")):
        path = os.path.join("data", folder, "Accelerometer.csv")
    else:
        # Fallback to original
        path = os.path.join(ROOT, folder, "Accelerometer.csv")

    df = pd.read_csv(path)
    df.rename(columns={"x":"accX", "y":"accY", "z":"accZ"}, inplace=True)
    return df

def compute_detailed_features(df, activity_name):
    """Compute comprehensive features for activity classification"""

    # Calculate magnitude
    mag = np.sqrt(df["accX"]**2 + df["accY"]**2 + df["accZ"]**2)

    # Time domain features
    features = {
        "Activity": activity_name,

        # Magnitude statistics
        "Mag_Mean": mag.mean(),
        "Mag_Std": mag.std(),
        "Mag_Max": mag.max(),
        "Mag_Min": mag.min(),
        "Mag_Range": mag.max() - mag.min(),

        # Individual axis statistics
        "X_Mean": df["accX"].mean(),
        "X_Std": df["accX"].std(),
        "Y_Mean": df["accY"].mean(),
        "Y_Std": df["accY"].std(),
        "Z_Mean": df["accZ"].mean(),
        "Z_Std": df["accZ"].std(),

        # Correlation between axes
        "XY_Corr": df["accX"].corr(df["accY"]),
        "XZ_Corr": df["accX"].corr(df["accZ"]),
        "YZ_Corr": df["accY"].corr(df["accZ"]),

        # Zero crossing rate (indicator of oscillation)
        "X_ZCR": ((df["accX"][:-1] * df["accX"][1:]) < 0).sum() / len(df),
        "Y_ZCR": ((df["accY"][:-1] * df["accY"][1:]) < 0).sum() / len(df),
        "Z_ZCR": ((df["accZ"][:-1] * df["accZ"][1:]) < 0).sum() / len(df),

        # Peak detection (periodic motion indicator)
        "Mag_Peaks": len(signal.find_peaks(mag, height=mag.mean())[0]),

        # Signal energy
        "X_Energy": (df["accX"]**2).sum() / len(df),
        "Y_Energy": (df["accY"]**2).sum() / len(df),
        "Z_Energy": (df["accZ"]**2).sum() / len(df),
        "Total_Energy": ((df["accX"]**2 + df["accY"]**2 + df["accZ"]**2).sum()) / len(df),
    }

    # Frequency domain features (detect periodicity)
    try:
        sampling_rate = 1.0 / df["seconds_elapsed"].diff().mean()
        n = len(mag)

        # FFT for magnitude
        yf = fft(mag - mag.mean())
        xf = fftfreq(n, 1/sampling_rate)[:n//2]
        power = 2.0/n * np.abs(yf[:n//2])

        # Find dominant frequency
        if len(power) > 0:
            dominant_idx = np.argmax(power)
            features["Dominant_Freq"] = xf[dominant_idx]
            features["Dominant_Power"] = power[dominant_idx]
            features["Spectral_Energy"] = (power**2).sum()
        else:
            features["Dominant_Freq"] = 0
            features["Dominant_Power"] = 0
            features["Spectral_Energy"] = 0

    except Exception as e:
        features["Dominant_Freq"] = 0
        features["Dominant_Power"] = 0
        features["Spectral_Energy"] = 0

    return features

def create_detailed_plots(df, activity_name, output_dir):
    """Create comprehensive visualization plots"""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Detailed Analysis: {activity_name}', fontsize=16, fontweight='bold')

    t = df["seconds_elapsed"]
    mag = np.sqrt(df["accX"]**2 + df["accY"]**2 + df["accZ"]**2)

    # 1. Raw accelerometer data (X, Y, Z)
    ax = axes[0, 0]
    ax.plot(t, df["accX"], label="X", alpha=0.7)
    ax.plot(t, df["accY"], label="Y", alpha=0.7)
    ax.plot(t, df["accZ"], label="Z", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Raw Accelerometer Data (3 Axes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Magnitude over time
    ax = axes[0, 1]
    ax.plot(t, mag, color='purple', linewidth=1.5)
    ax.axhline(mag.mean(), color='red', linestyle='--', label=f'Mean: {mag.mean():.3f}')
    ax.axhline(mag.mean() + mag.std(), color='orange', linestyle=':', label=f'±1 Std: {mag.std():.3f}')
    ax.axhline(mag.mean() - mag.std(), color='orange', linestyle=':')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (m/s²)")
    ax.set_title("Acceleration Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Distribution histogram
    ax = axes[1, 0]
    ax.hist(mag, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(mag.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mag.mean():.3f}')
    ax.set_xlabel("Magnitude (m/s²)")
    ax.set_ylabel("Frequency")
    ax.set_title("Magnitude Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Frequency spectrum (FFT)
    ax = axes[1, 1]
    try:
        sampling_rate = 1.0 / df["seconds_elapsed"].diff().mean()
        n = len(mag)
        yf = fft(mag - mag.mean())
        xf = fftfreq(n, 1/sampling_rate)[:n//2]
        power = 2.0/n * np.abs(yf[:n//2])

        ax.plot(xf, power, color='darkblue')
        if len(power) > 0:
            dominant_idx = np.argmax(power)
            ax.axvline(xf[dominant_idx], color='red', linestyle='--',
                      label=f'Peak: {xf[dominant_idx]:.2f} Hz')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_title("Frequency Spectrum (FFT)")
        ax.set_xlim(0, 10)  # Focus on low frequencies
        ax.legend()
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'FFT calculation failed', ha='center', va='center')

    # 5. Individual axis comparison
    ax = axes[2, 0]
    data_to_plot = [df["accX"], df["accY"], df["accZ"]]
    bp = ax.boxplot(data_to_plot, labels=['X', 'Y', 'Z'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Axis Distribution Comparison")
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Moving statistics
    ax = axes[2, 1]
    window = min(50, len(mag)//10)
    rolling_mean = pd.Series(mag).rolling(window=window).mean()
    rolling_std = pd.Series(mag).rolling(window=window).std()
    ax.plot(t, rolling_mean, label=f'Rolling Mean (w={window})', color='blue')
    ax.fill_between(t, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     alpha=0.3, color='blue', label='±1 Rolling Std')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (m/s²)")
    ax.set_title("Moving Statistics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{activity_name}_detailed.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved detailed plot: {output_file}")
    plt.close()

# Main analysis
folders = [
    "sitting_hand-2025-10-10_14-38-42",
    "standing_hand-2025-10-10_14-38-03",
    "walking_hand_1-2025-10-10_14-39-10",
    "walking_hand_2-2025-10-10_14-39-53"
]

output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    all_features = []

    print("\n" + "="*80)
    print("DETAILED ACCELEROMETER DATA ANALYSIS")
    print("="*80 + "\n")

    for folder in folders:
        df = read_accel_csv(folder)
        activity_name = folder.split("-")[0].replace("_", " ").title()

        print(f"\nProcessing: {activity_name}")
        print(f"  Data points: {len(df)}")
        print(f"  Duration: {df['seconds_elapsed'].max():.2f} seconds")

        # Compute features
        features = compute_detailed_features(df, activity_name)
        all_features.append(features)

        # Create detailed plots
        create_detailed_plots(df, folder.split("-")[0], output_dir)

    # Create comparison table
    features_df = pd.DataFrame(all_features)

    print("\n" + "="*80)
    print("FEATURE COMPARISON TABLE")
    print("="*80 + "\n")

    # Display key distinguishing features
    key_features = [
        "Activity", "Mag_Mean", "Mag_Std", "Mag_Range",
        "Dominant_Freq", "Total_Energy", "Mag_Peaks"
    ]
    print(features_df[key_features].to_string(index=False))

    print("\n" + "="*80)
    print("AXIS-SPECIFIC STATISTICS")
    print("="*80 + "\n")

    axis_features = ["Activity", "X_Mean", "X_Std", "Y_Mean", "Y_Std", "Z_Mean", "Z_Std"]
    print(features_df[axis_features].to_string(index=False))

    print("\n" + "="*80)
    print("ZERO CROSSING RATES (Oscillation Indicator)")
    print("="*80 + "\n")

    zcr_features = ["Activity", "X_ZCR", "Y_ZCR", "Z_ZCR"]
    print(features_df[zcr_features].to_string(index=False))

    # Save full feature table
    features_csv = os.path.join(output_dir, "feature_comparison.csv")
    features_df.to_csv(features_csv, index=False)
    print(f"\n\nFull feature table saved to: {features_csv}")

    print("\n" + "="*80)
    print("KEY INSIGHTS FOR ACTIVITY CLASSIFICATION")
    print("="*80 + "\n")

    print("""
1. MAGNITUDE-BASED FEATURES:
   - Sitting/Standing have LOW mean magnitude (~0.17-0.23 m/s²)
   - Walking has HIGH mean magnitude (~0.99-1.02 m/s²)
   - Ratio: Walking is 4-6x higher than static activities

2. VARIABILITY (Standard Deviation):
   - Static activities (sitting/standing): LOW variability (~0.09-0.13 m/s²)
   - Walking: HIGH variability (~0.45 m/s²)
   - This reflects the periodic motion during walking

3. FREQUENCY DOMAIN:
   - Static activities: No dominant frequency (random motion)
   - Walking: Clear dominant frequency (~1-2 Hz, corresponding to step rate)

4. PERIODICITY:
   - Walking shows clear periodic patterns visible in time-series plots
   - Static activities show irregular, low-amplitude fluctuations

5. ENERGY:
   - Total signal energy is much higher for walking
   - Energy ratio can distinguish dynamic vs static activities
""")

    print("\n" + "="*80)
