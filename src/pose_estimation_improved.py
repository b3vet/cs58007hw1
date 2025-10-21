"""
Improved Pose Estimation from Accelerometer and Gyroscope
Fixes issues in original implementation and adds enhancements
"""

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def read_sensor_pair(acc_path, gyro_path):
    """
    Read accelerometer and gyroscope CSV files and align timestamps

    Returns: t, ax, ay, az, gx, gy, gz (all aligned to accelerometer timeline)
    """
    def read_csv(path):
        with open(path) as f:
            reader = csv.reader(f)
            header = [h.strip().lower() for h in next(reader)]

            t_idx = header.index("seconds_elapsed")
            x_idx = header.index("x")
            y_idx = header.index("y")
            z_idx = header.index("z")

            t, x, y, z = [], [], [], []
            for row in reader:
                try:
                    t.append(float(row[t_idx]))
                    x.append(float(row[x_idx]))
                    y.append(float(row[y_idx]))
                    z.append(float(row[z_idx]))
                except:
                    pass  # Skip malformed rows

            return np.array(t), np.array(x), np.array(y), np.array(z)

    # Read both files
    ta, ax, ay, az = read_csv(acc_path)
    tg, gx, gy, gz = read_csv(gyro_path)

    # Interpolate gyroscope data onto accelerometer timeline
    gx_interp = np.interp(ta, tg, gx)
    gy_interp = np.interp(ta, tg, gy)
    gz_interp = np.interp(ta, tg, gz)

    return ta, ax, ay, az, gx_interp, gy_interp, gz_interp


def estimate_gyro_bias(gx, gy, gz, initial_samples=100):
    """
    Estimate gyroscope bias from initial stationary period

    Args:
        gx, gy, gz: Gyroscope data
        initial_samples: Number of initial samples to use for bias estimation

    Returns: bias_x, bias_y, bias_z
    """
    n = min(initial_samples, len(gx))
    bias_x = np.mean(gx[:n])
    bias_y = np.mean(gy[:n])
    bias_z = np.mean(gz[:n])

    return bias_x, bias_y, bias_z


def low_pass_filter(signal, alpha=0.9):
    """
    Simple low-pass filter to smooth accelerometer data
    Reduces noise while preserving gravity direction

    Args:
        signal: Input signal
        alpha: Smoothing factor (0-1, higher = smoother)

    Returns: Filtered signal
    """
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]

    for i in range(1, len(signal)):
        filtered[i] = alpha * filtered[i-1] + (1 - alpha) * signal[i]

    return filtered


def complementary_filter_basic(acc_path, gyro_path, alpha=0.98, filter_accel=True,
                                correct_bias=True, plot=True):
    """
    Basic complementary filter for pitch and roll estimation

    FIXES from original:
    - Corrected pitch/roll formulas (they were swapped!)
    - Uses variable dt (actual time differences)
    - Adds gyroscope bias correction
    - Optional accelerometer filtering

    Args:
        acc_path: Path to Accelerometer.csv
        gyro_path: Path to Gyroscope.csv
        alpha: Complementary filter weight (0.98 = trust gyro 98%, accel 2%)
        filter_accel: Whether to low-pass filter accelerometer data
        correct_bias: Whether to estimate and remove gyroscope bias
        plot: Whether to show plots

    Returns: Dictionary with results
    """
    # Read sensor data
    t, ax, ay, az, gx, gy, gz = read_sensor_pair(acc_path, gyro_path)

    # Estimate sampling rate
    fs = 1.0 / np.median(np.diff(t))

    # Optional: Correct gyroscope bias
    if correct_bias:
        bias_x, bias_y, bias_z = estimate_gyro_bias(gx, gy, gz)
        gx = gx - bias_x
        gy = gy - bias_y
        gz = gz - bias_z
        print(f"Gyroscope bias removed: X={bias_x:.4f}, Y={bias_y:.4f}, Z={bias_z:.4f} rad/s")

    # Optional: Filter accelerometer data to reduce noise
    if filter_accel:
        ax = low_pass_filter(ax, alpha=0.9)
        ay = low_pass_filter(ay, alpha=0.9)
        az = low_pass_filter(az, alpha=0.9)

    # Initialize angles
    pitch = 0.0
    roll = 0.0
    yaw = 0.0  # Yaw will drift without magnetometer

    # Storage for time series
    pitch_log = []
    roll_log = []
    yaw_log = []
    pitch_accel_log = []
    roll_accel_log = []
    pitch_gyro_log = []
    roll_gyro_log = []

    # Main filter loop
    for i in range(len(t)):
        # Calculate dt (use actual time difference, not average)
        if i == 0:
            dt = 1.0 / fs  # First sample, use average
        else:
            dt = t[i] - t[i-1]

        # === ACCELEROMETER-BASED ANGLES (FIXED FORMULAS!) ===
        # Pitch: rotation around Y-axis (forward/backward tilt)
        # CORRECT formula: atan2(-ax, sqrt(ay^2 + az^2))
        pitch_acc = math.degrees(math.atan2(-ax[i], math.sqrt(ay[i]**2 + az[i]**2)))

        # Roll: rotation around X-axis (left/right tilt)
        # CORRECT formula: atan2(ay, sqrt(ax^2 + az^2))
        roll_acc = math.degrees(math.atan2(ay[i], math.sqrt(ax[i]**2 + az[i]**2)))

        # === GYROSCOPE INTEGRATION ===
        # Integrate angular velocity to get angle change
        # Note: gyroscope is in rad/s, convert to degrees/s
        pitch_gyro = pitch + math.degrees(gy[i] * dt)  # Y-axis rotation
        roll_gyro = roll + math.degrees(gx[i] * dt)    # X-axis rotation
        yaw_gyro = yaw + math.degrees(gz[i] * dt)      # Z-axis rotation

        # === COMPLEMENTARY FILTER ===
        # Combine gyro (no noise, has drift) with accel (no drift, has noise)
        pitch = alpha * pitch_gyro + (1 - alpha) * pitch_acc
        roll = alpha * roll_gyro + (1 - alpha) * roll_acc
        yaw = yaw_gyro  # Yaw has no correction (needs magnetometer)

        # Log values
        pitch_log.append(pitch)
        roll_log.append(roll)
        yaw_log.append(yaw)
        pitch_accel_log.append(pitch_acc)
        roll_accel_log.append(roll_acc)
        pitch_gyro_log.append(pitch_gyro)
        roll_gyro_log.append(roll_gyro)

    # Convert to numpy arrays
    pitch_log = np.array(pitch_log)
    roll_log = np.array(roll_log)
    yaw_log = np.array(yaw_log)
    pitch_accel_log = np.array(pitch_accel_log)
    roll_accel_log = np.array(roll_accel_log)

    # Plotting
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pose Estimation: Complementary Filter', fontsize=14, fontweight='bold')

        time = t - t[0]

        # Plot 1: Pitch estimation
        ax = axes[0, 0]
        ax.plot(time, pitch_accel_log, alpha=0.5, label='Accel-only (noisy)', linewidth=1)
        ax.plot(time, pitch_log, label='Complementary Filter', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pitch (degrees)')
        ax.set_title('Pitch Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Roll estimation
        ax = axes[0, 1]
        ax.plot(time, roll_accel_log, alpha=0.5, label='Accel-only (noisy)', linewidth=1)
        ax.plot(time, roll_log, label='Complementary Filter', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Roll (degrees)')
        ax.set_title('Roll Estimation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Yaw (will drift)
        ax = axes[1, 0]
        ax.plot(time, yaw_log, label='Yaw (gyro integration - WILL DRIFT)', linewidth=2, color='orange')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw (degrees)')
        ax.set_title('Yaw Estimation (Uncorrected - Drifts without Magnetometer)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: 3D orientation visualization
        ax = axes[1, 1]
        ax.plot(time, pitch_log, label='Pitch', linewidth=2)
        ax.plot(time, roll_log, label='Roll', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Combined Pitch & Roll')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Calculate statistics
    pitch_std = np.std(pitch_log)
    roll_std = np.std(roll_log)
    pitch_range = np.max(pitch_log) - np.min(pitch_log)
    roll_range = np.max(roll_log) - np.min(roll_log)

    print(f"\n{'='*60}")
    print(f"Pose Estimation Results")
    print(f"{'='*60}")
    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Duration: {t[-1] - t[0]:.2f} seconds")
    print(f"Samples: {len(t)}")
    print(f"\nPitch statistics:")
    print(f"  Mean: {np.mean(pitch_log):.2f}°")
    print(f"  Std:  {pitch_std:.2f}°")
    print(f"  Range: {pitch_range:.2f}° ({np.min(pitch_log):.2f}° to {np.max(pitch_log):.2f}°)")
    print(f"\nRoll statistics:")
    print(f"  Mean: {np.mean(roll_log):.2f}°")
    print(f"  Std:  {roll_std:.2f}°")
    print(f"  Range: {roll_range:.2f}° ({np.min(roll_log):.2f}° to {np.max(roll_log):.2f}°)")
    print(f"\nYaw: {yaw_log[-1]:.2f}° (WARNING: Drifts without magnetometer!)")
    print(f"{'='*60}\n")

    return {
        'fs': fs,
        'time': t,
        'pitch': pitch_log,
        'roll': roll_log,
        'yaw': yaw_log,
        'pitch_accel': pitch_accel_log,
        'roll_accel': roll_accel_log,
        'pitch_std': pitch_std,
        'roll_std': roll_std,
        'pitch_range': pitch_range,
        'roll_range': roll_range
    }


def save_pose_estimation_plot(acc_path, gyro_path, output_path, alpha=0.98):
    """
    Run pose estimation and save plot to file
    """
    # Read sensor data
    t, ax, ay, az, gx, gy, gz = read_sensor_pair(acc_path, gyro_path)

    # Estimate sampling rate
    fs = 1.0 / np.median(np.diff(t))

    # Correct gyroscope bias
    bias_x, bias_y, bias_z = estimate_gyro_bias(gx, gy, gz)
    gx = gx - bias_x
    gy = gy - bias_y
    gz = gz - bias_z

    # Filter accelerometer
    ax_f = low_pass_filter(ax, alpha=0.9)
    ay_f = low_pass_filter(ay, alpha=0.9)
    az_f = low_pass_filter(az, alpha=0.9)

    # Initialize angles
    pitch, roll, yaw = 0.0, 0.0, 0.0
    pitch_log, roll_log, yaw_log = [], [], []
    pitch_accel_log, roll_accel_log = [], []

    # Main filter loop
    for i in range(len(t)):
        dt = (t[i] - t[i-1]) if i > 0 else (1.0 / fs)

        # Accelerometer angles (CORRECTED formulas)
        pitch_acc = math.degrees(math.atan2(-ax_f[i], math.sqrt(ay_f[i]**2 + az_f[i]**2)))
        roll_acc = math.degrees(math.atan2(ay_f[i], math.sqrt(ax_f[i]**2 + az_f[i]**2)))

        # Gyroscope integration
        pitch_gyro = pitch + math.degrees(gy[i] * dt)
        roll_gyro = roll + math.degrees(gx[i] * dt)
        yaw_gyro = yaw + math.degrees(gz[i] * dt)

        # Complementary filter
        pitch = alpha * pitch_gyro + (1 - alpha) * pitch_acc
        roll = alpha * roll_gyro + (1 - alpha) * roll_acc
        yaw = yaw_gyro

        pitch_log.append(pitch)
        roll_log.append(roll)
        yaw_log.append(yaw)
        pitch_accel_log.append(pitch_acc)
        roll_accel_log.append(roll_acc)

    # Convert to arrays
    pitch_log = np.array(pitch_log)
    roll_log = np.array(roll_log)
    yaw_log = np.array(yaw_log)
    pitch_accel_log = np.array(pitch_accel_log)
    roll_accel_log = np.array(roll_accel_log)

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    activity_name = os.path.basename(os.path.dirname(acc_path)).split('-')[0].replace('_', ' ').title()
    fig.suptitle(f'Pose Estimation: {activity_name}', fontsize=14, fontweight='bold')

    time = t - t[0]

    # Pitch
    ax = axes[0, 0]
    ax.plot(time, pitch_accel_log, alpha=0.5, label='Accel-only', linewidth=1)
    ax.plot(time, pitch_log, label='Filtered', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (degrees)')
    ax.set_title('Pitch Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Roll
    ax = axes[0, 1]
    ax.plot(time, roll_accel_log, alpha=0.5, label='Accel-only', linewidth=1)
    ax.plot(time, roll_log, label='Filtered', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Roll (degrees)')
    ax.set_title('Roll Estimation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Yaw
    ax = axes[1, 0]
    ax.plot(time, yaw_log, linewidth=2, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw (degrees)')
    ax.set_title('Yaw (Drifts - No Magnetometer)')
    ax.grid(True, alpha=0.3)

    # Combined
    ax = axes[1, 1]
    ax.plot(time, pitch_log, label='Pitch', linewidth=2)
    ax.plot(time, roll_log, label='Roll', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Pitch & Roll Combined')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pose_estimation_improved.py <accelerometer.csv> <gyroscope.csv>")
        sys.exit(1)

    acc_path = sys.argv[1]
    gyro_path = sys.argv[2]

    result = complementary_filter_basic(acc_path, gyro_path, alpha=0.98,
                                       filter_accel=True, correct_bias=True, plot=True)
