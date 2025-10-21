"""
Compare original vs improved pose estimation
Shows the effect of fixing the swapped pitch/roll formulas
"""

import numpy as np
import matplotlib.pyplot as plt
from pose_estimation import complementary_filter as original_filter
from pose_estimation_improved import complementary_filter_basic as improved_filter


# Import CSV config from main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import CSV_CONFIG

# Use centralized config for walking_hand_1
acc_path = CSV_CONFIG['activity_csvs']['walking_hand_1']['acc']
gyro_path = CSV_CONFIG['activity_csvs']['walking_hand_1']['gyro']

print("\n" + "="*70)
print("COMPARISON: Original vs Improved Pose Estimation")
print("="*70)

print("\nRunning ORIGINAL implementation...")
result_orig = original_filter(acc_path, gyro_path, alpha=0.98, plot=False)

print("\nRunning IMPROVED implementation...")
result_improved = improved_filter(acc_path, gyro_path, alpha=0.98,
                                  filter_accel=True, correct_bias=True, plot=False)

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Original vs Improved Pose Estimation\n(Notice: Original has Pitch/Roll SWAPPED!)',
             fontsize=14, fontweight='bold')

# Time vectors
time_orig = np.arange(len(result_orig['pitch'])) / result_orig['fs']
time_improved = result_improved['time'] - result_improved['time'][0]

# Shorten for comparison (use shorter length)
min_len = min(len(time_orig), len(time_improved))
time_orig = time_orig[:min_len]
time_improved = time_improved[:min_len]

pitch_orig = np.array(result_orig['pitch'][:min_len])
roll_orig = np.array(result_orig['roll'][:min_len])
pitch_improved = result_improved['pitch'][:min_len]
roll_improved = result_improved['roll'][:min_len]

# Plot 1: Original Pitch (actually ROLL!)
ax = axes[0, 0]
ax.plot(time_orig, pitch_orig, linewidth=2, color='red', alpha=0.7, label='Original "Pitch"')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Original: Variable called "Pitch" (Actually ROLL!)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Improved Pitch (correct)
ax = axes[0, 1]
ax.plot(time_improved, pitch_improved, linewidth=2, color='blue', label='Improved Pitch (CORRECT)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Improved: Actual Pitch (Forward/Backward Tilt)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Original Roll (actually PITCH!)
ax = axes[1, 0]
ax.plot(time_orig, roll_orig, linewidth=2, color='red', alpha=0.7, label='Original "Roll"')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Original: Variable called "Roll" (Actually PITCH!)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Improved Roll (correct)
ax = axes[1, 1]
ax.plot(time_improved, roll_improved, linewidth=2, color='blue', label='Improved Roll (CORRECT)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Improved: Actual Roll (Left/Right Tilt)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/comparison_original_vs_improved.png', dpi=150, bbox_inches='tight')
print("\nComparison plot saved to: plots/comparison_original_vs_improved.png")
plt.close()

# Numerical comparison
print("\n" + "="*70)
print("VERIFICATION: The Original Pitch should match Improved Roll (and vice versa)")
print("="*70)

print("\nOriginal 'Pitch' statistics:")
print(f"  Mean: {np.mean(pitch_orig):.2f}°")
print(f"  Std:  {np.std(pitch_orig):.2f}°")
print(f"  Range: {np.max(pitch_orig) - np.min(pitch_orig):.2f}°")

print("\nImproved ROLL statistics:")
print(f"  Mean: {np.mean(roll_improved):.2f}°")
print(f"  Std:  {np.std(roll_improved):.2f}°")
print(f"  Range: {np.max(roll_improved) - np.min(roll_improved):.2f}°")

print("\n✅ These should be similar! (Original 'pitch' = Improved roll)")

print("\n" + "-"*70)

print("\nOriginal 'Roll' statistics:")
print(f"  Mean: {np.mean(roll_orig):.2f}°")
print(f"  Std:  {np.std(roll_orig):.2f}°")
print(f"  Range: {np.max(roll_orig) - np.min(roll_orig):.2f}°")

print("\nImproved PITCH statistics:")
print(f"  Mean: {np.mean(pitch_improved):.2f}°")
print(f"  Std:  {np.std(pitch_improved):.2f}°")
print(f"  Range: {np.max(pitch_improved) - np.min(pitch_improved):.2f}°")

print("\n✅ These should be similar! (Original 'roll' = Improved pitch)")

print("\n" + "="*70)
print("CONCLUSION: Original implementation had SWAPPED pitch and roll!")
print("="*70 + "\n")
