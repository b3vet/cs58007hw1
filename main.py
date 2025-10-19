#!/usr/bin/env python3
"""
Main entry point for Human Activity Recognition and Pose Estimation project

This script orchestrates all three parts:
  Part 1: Activity Classification and Feature Analysis
  Part 2: Step Counting from Accelerometer Data
  Part 3: Pose Estimation from Accelerometer and Gyroscope

Usage:
  python main.py --all                    # Run all parts on all activities
  python main.py --part1                  # Run only activity analysis
  python main.py --part2                  # Run only step counting
  python main.py --part3                  # Run only pose estimation
  python main.py --activity walking_hand_1  # Run all parts on specific activity
"""

import os
import sys
import argparse
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directories
DATA_DIR = "data"
PLOTS_DIR = "plots"

# Available activities
ACTIVITIES = [
    "sitting_hand-2025-10-10_14-38-42",
    "standing_hand-2025-10-10_14-38-03",
    "walking_hand_1-2025-10-10_14-39-10",
    "walking_hand_2-2025-10-10_14-39-53"
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_banner(text, char='=', width=80):
    """Print a formatted banner"""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_section(text, char='-', width=80):
    """Print a section header"""
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}\n")


def get_activity_paths(activity_name):
    """Get full paths for activity data files"""
    activity_dir = os.path.join(DATA_DIR, activity_name)
    return {
        'name': activity_name.split('-')[0],
        'acc': os.path.join(activity_dir, "Accelerometer.csv"),
        'gyro': os.path.join(activity_dir, "Gyroscope.csv"),
        'dir': activity_dir
    }


# ============================================================================
# PART 1: ACTIVITY CLASSIFICATION
# ============================================================================

def run_part1(activities=None, save_plots=True):
    """
    Part 1: Activity Classification and Feature Analysis

    Analyzes accelerometer data to identify distinguishing features
    between sitting, standing, and walking activities.
    """
    print_banner("PART 1: ACTIVITY CLASSIFICATION & FEATURE ANALYSIS")

    try:
        import sys
        sys.path.insert(0, 'src')
        from detailed_analysis import (
            read_accel_csv,
            compute_detailed_features,
            create_detailed_plots
        )
        import pandas as pd

        if activities is None:
            activities = ACTIVITIES

        all_features = []

        for activity in activities:
            paths = get_activity_paths(activity)
            activity_name = paths['name']

            print_section(f"Analyzing: {activity_name.replace('_', ' ').title()}")

            try:
                # Read data
                df = read_accel_csv(activity)

                print(f"Data points: {len(df)}")
                print(f"Duration: {df['seconds_elapsed'].max():.2f} seconds")

                # Compute features
                features = compute_detailed_features(df, activity_name.title())
                all_features.append(features)

                # Create plots if requested
                if save_plots:
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    output_file = os.path.join(PLOTS_DIR, f"{activity_name}_detailed.png")
                    create_detailed_plots(df, activity_name, PLOTS_DIR)

                # Print key features
                print(f"\nKey Features:")
                print(f"  Magnitude Mean:    {features['Mag_Mean']:.3f} m/s²")
                print(f"  Magnitude Std:     {features['Mag_Std']:.3f} m/s²")
                print(f"  Dominant Freq:     {features['Dominant_Freq']:.3f} Hz")
                print(f"  Total Energy:      {features['Total_Energy']:.3f}")
                print(f"  Peak Count:        {features['Mag_Peaks']}")

            except Exception as e:
                print(f"❌ Error processing {activity_name}: {e}")
                continue

        # Print comparison table
        if len(all_features) > 1:
            print_section("FEATURE COMPARISON")

            features_df = pd.DataFrame(all_features)
            key_features = [
                "Activity", "Mag_Mean", "Mag_Std", "Mag_Range",
                "Dominant_Freq", "Total_Energy", "Mag_Peaks"
            ]
            print(features_df[key_features].to_string(index=False))

            # Save feature table
            output_csv = os.path.join(PLOTS_DIR, "feature_comparison.csv")
            features_df.to_csv(output_csv, index=False)
            print(f"\n✓ Feature table saved to: {output_csv}")

        print_section("PART 1 COMPLETE")
        print("✓ Activity analysis complete")
        print(f"✓ Processed {len(all_features)} activities")
        if save_plots:
            print(f"✓ Plots saved to {PLOTS_DIR}/")

        return all_features

    except ImportError as e:
        print(f"❌ Error importing Part 1 modules: {e}")
        print("Make sure src/detailed_analysis.py exists")
        return None


# ============================================================================
# PART 2: STEP COUNTING
# ============================================================================

def run_part2(activities=None, save_plots=True):
    """
    Part 2: Step Counting from Accelerometer Data

    Counts steps using multiple methods: peak detection, autocorrelation,
    and zero crossings.
    """
    print_banner("PART 2: STEP COUNTING FROM ACCELEROMETER DATA")

    try:
        import sys
        sys.path.insert(0, 'src')
        from stepcount_enhanced import count_steps_from_csv
        from visualize_steps import visualize_step_detection

        if activities is None:
            # Only run on walking activities for step counting
            activities = [a for a in ACTIVITIES if 'walking' in a.lower()]

        results = []

        for activity in activities:
            paths = get_activity_paths(activity)
            activity_name = paths['name']

            # Skip non-walking activities
            if 'walking' not in activity_name.lower():
                continue

            print_section(f"Step Counting: {activity_name.replace('_', ' ').title()}")

            try:
                # Count steps using combined method
                result = count_steps_from_csv(
                    paths['acc'],
                    method='combined',
                    verbose=False
                )

                results.append({
                    'activity': activity_name,
                    'steps': result['steps'],
                    'duration': result['duration_sec'],
                    'cadence': result['cadence_spm']
                })

                # Print results
                print(f"Sampling rate:     {result['sampling_rate']:.2f} Hz")
                print(f"Duration:          {result['duration_sec']:.2f} seconds")
                print(f"Detected steps:    {result['steps']}")
                print(f"Cadence:           {result['cadence_spm']:.1f} steps/minute")

                # Method breakdown
                if 'details' in result:
                    details = result['details']
                    print(f"\nMethod Breakdown:")
                    print(f"  Peak Detection:    {details.get('method1_steps', 'N/A')} steps")
                    print(f"  Autocorrelation:   {details.get('method2_steps', 'N/A')} steps")
                    print(f"  Zero Crossings:    {details.get('method3_steps', 'N/A')} steps")
                    print(f"  Combined (Median): {result['steps']} steps")

                # Generate visualization
                if save_plots:
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    output_path = os.path.join(PLOTS_DIR, f"{activity_name}_step_analysis.png")
                    visualize_step_detection(paths['acc'], output_path)

            except Exception as e:
                print(f"❌ Error processing {activity_name}: {e}")
                continue

        # Print summary
        if results:
            print_section("STEP COUNTING SUMMARY")
            print(f"{'Activity':<20} {'Steps':<10} {'Duration':<12} {'Cadence'}")
            print("-" * 60)
            for r in results:
                print(f"{r['activity']:<20} {r['steps']:<10} {r['duration']:.2f}s {' '*6} {r['cadence']:.1f} steps/min")

        print_section("PART 2 COMPLETE")
        print(f"✓ Step counting complete for {len(results)} walking activities")
        if save_plots:
            print(f"✓ Visualizations saved to {PLOTS_DIR}/")

        return results

    except ImportError as e:
        print(f"❌ Error importing Part 2 modules: {e}")
        print("Make sure src/stepcount_enhanced.py and src/visualize_steps.py exist")
        return None


# ============================================================================
# PART 3: POSE ESTIMATION
# ============================================================================

def run_part3(activities=None, save_plots=True):
    """
    Part 3: Pose Estimation from Accelerometer and Gyroscope

    Estimates device orientation (pitch, roll, yaw) using complementary filter.
    """
    print_banner("PART 3: POSE ESTIMATION (ACCELEROMETER + GYROSCOPE)")

    try:
        import sys
        sys.path.insert(0, 'src')
        from pose_estimation_improved import save_pose_estimation_plot
        import pandas as pd
        import numpy as np
        from pose_estimation_improved import (
            read_sensor_pair,
            estimate_gyro_bias,
            low_pass_filter
        )
        import math

        # Define estimate_sampling_rate locally
        def estimate_sampling_rate(timestamps):
            dt = np.diff(timestamps)
            dt = dt[dt > 0]
            median_dt = np.median(dt)
            return 1.0 / median_dt if median_dt > 0 else 100.0

        if activities is None:
            activities = ACTIVITIES

        results = []

        for activity in activities:
            paths = get_activity_paths(activity)
            activity_name = paths['name']

            print_section(f"Pose Estimation: {activity_name.replace('_', ' ').title()}")

            try:
                # Read sensor data
                t, ax, ay, az, gx, gy, gz = read_sensor_pair(paths['acc'], paths['gyro'])

                # Basic info
                fs = estimate_sampling_rate(t)
                duration = t[-1] - t[0]

                print(f"Sampling rate:     {fs:.2f} Hz")
                print(f"Duration:          {duration:.2f} seconds")
                print(f"Data points:       {len(t)}")

                # Estimate gyro bias
                bias_x, bias_y, bias_z = estimate_gyro_bias(gx, gy, gz)
                print(f"\nGyroscope bias:    X={bias_x:.4f}, Y={bias_y:.4f}, Z={bias_z:.4f} rad/s")

                # Correct bias
                gx = gx - bias_x
                gy = gy - bias_y
                gz = gz - bias_z

                # Filter accelerometer
                ax_f = low_pass_filter(ax, alpha=0.9)
                ay_f = low_pass_filter(ay, alpha=0.9)
                az_f = low_pass_filter(az, alpha=0.9)

                # Complementary filter
                alpha = 0.98
                pitch, roll, yaw = 0.0, 0.0, 0.0
                pitch_log, roll_log, yaw_log = [], [], []

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

                # Calculate statistics
                pitch_arr = np.array(pitch_log)
                roll_arr = np.array(roll_log)
                yaw_arr = np.array(yaw_log)

                result = {
                    'activity': activity_name,
                    'pitch_mean': np.mean(pitch_arr),
                    'pitch_std': np.std(pitch_arr),
                    'pitch_range': np.max(pitch_arr) - np.min(pitch_arr),
                    'roll_mean': np.mean(roll_arr),
                    'roll_std': np.std(roll_arr),
                    'roll_range': np.max(roll_arr) - np.min(roll_arr),
                    'yaw_drift': yaw_arr[-1]
                }
                results.append(result)

                # Print statistics
                print(f"\nPitch (Forward/Backward Tilt):")
                print(f"  Mean:  {result['pitch_mean']:7.2f}°")
                print(f"  Std:   {result['pitch_std']:7.2f}°")
                print(f"  Range: {result['pitch_range']:7.2f}° ({np.min(pitch_arr):.2f}° to {np.max(pitch_arr):.2f}°)")

                print(f"\nRoll (Left/Right Tilt):")
                print(f"  Mean:  {result['roll_mean']:7.2f}°")
                print(f"  Std:   {result['roll_std']:7.2f}°")
                print(f"  Range: {result['roll_range']:7.2f}° ({np.min(roll_arr):.2f}° to {np.max(roll_arr):.2f}°)")

                print(f"\nYaw (Rotation):")
                print(f"  Drift: {result['yaw_drift']:7.2f}° (⚠️  No magnetometer - will drift)")

                # Generate visualization
                if save_plots:
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    output_path = os.path.join(PLOTS_DIR, f"{activity_name}_pose_estimation.png")
                    save_pose_estimation_plot(paths['acc'], paths['gyro'], output_path, alpha=0.98)

            except Exception as e:
                print(f"❌ Error processing {activity_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Print summary
        if results:
            print_section("POSE ESTIMATION SUMMARY")
            print(f"{'Activity':<20} {'Pitch Std':<12} {'Roll Std':<12} {'Yaw Drift'}")
            print("-" * 70)
            for r in results:
                print(f"{r['activity']:<20} {r['pitch_std']:7.2f}° {' '*4} {r['roll_std']:7.2f}° {' '*4} {r['yaw_drift']:7.2f}°")

        print_section("PART 3 COMPLETE")
        print(f"✓ Pose estimation complete for {len(results)} activities")
        if save_plots:
            print(f"✓ Visualizations saved to {PLOTS_DIR}/")

        return results

    except ImportError as e:
        print(f"❌ Error importing Part 3 modules: {e}")
        print("Make sure src/pose_estimation_improved.py exists")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Human Activity Recognition and Pose Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run all parts (default)
  python main.py --all                    # Run all parts (explicit)
  python main.py --part1                  # Run only Part 1
  python main.py --part2                  # Run only Part 2
  python main.py --part3                  # Run only Part 3
  python main.py --activity walking_hand_1  # Run all parts on specific activity
  python main.py --part2 --no-plots       # Run Part 2 without saving plots
        """
    )

    # Part selection
    parser.add_argument('--all', action='store_true',
                       help='Run all three parts (default if no part specified)')
    parser.add_argument('--part1', action='store_true',
                       help='Run Part 1: Activity Classification')
    parser.add_argument('--part2', action='store_true',
                       help='Run Part 2: Step Counting')
    parser.add_argument('--part3', action='store_true',
                       help='Run Part 3: Pose Estimation')

    # Activity selection
    parser.add_argument('--activity', type=str,
                       help='Run on specific activity (e.g., walking_hand_1)')

    # Options
    parser.add_argument('--no-plots', action='store_true',
                       help='Do not save plots')

    args = parser.parse_args()

    # Determine which parts to run
    # If no specific part is selected, run all parts by default
    run_all = args.all or not (args.part1 or args.part2 or args.part3)
    run_p1 = run_all or args.part1
    run_p2 = run_all or args.part2
    run_p3 = run_all or args.part3

    # Determine which activities to process
    if args.activity:
        # Find matching activity
        matching = [a for a in ACTIVITIES if args.activity in a]
        if not matching:
            print(f"❌ Activity '{args.activity}' not found!")
            print(f"Available activities:")
            for a in ACTIVITIES:
                print(f"  - {a.split('-')[0]}")
            sys.exit(1)
        activities = matching
    else:
        activities = ACTIVITIES

    save_plots = not args.no_plots

    # Print header
    print_banner("HUMAN ACTIVITY RECOGNITION & POSE ESTIMATION", char='=', width=80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Activities to process: {len(activities)}")
    print(f"Parts to run: ", end='')
    parts = []
    if run_p1:
        parts.append("Part 1 (Activity Classification)")
    if run_p2:
        parts.append("Part 2 (Step Counting)")
    if run_p3:
        parts.append("Part 3 (Pose Estimation)")
    print(", ".join(parts))
    print(f"Save plots: {save_plots}")

    # Run requested parts
    try:
        if run_p1:
            result_p1 = run_part1(activities, save_plots)

        if run_p2:
            result_p2 = run_part2(activities, save_plots)

        if run_p3:
            result_p3 = run_part3(activities, save_plots)

        # Final summary
        print_banner("ALL TASKS COMPLETE", char='=', width=80)
        print("✓ All requested parts have been executed successfully")
        print(f"✓ Results saved to: {PLOTS_DIR}/")
        print("\nFor detailed documentation, see:")
        print("  - docs/ACTIVITY_ANALYSIS.md")
        print("  - docs/STEP_COUNTING_EXPLANATION.md")
        print("  - docs/POSE_ESTIMATION_REPORT.md")
        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
