"""
Test pose estimation on all recorded activities
"""

import os
from pose_estimation_improved import save_pose_estimation_plot

# Activity folders
activities = [
    "sitting_hand-2025-10-10_14-38-42",
    "standing_hand-2025-10-10_14-38-03",
    "walking_hand_1-2025-10-10_14-39-10",
    "walking_hand_2-2025-10-10_14-39-53"
]

base_dir = "../data"
output_dir = "../plots"

os.makedirs(output_dir, exist_ok=True)

for activity in activities:
    activity_name = activity.split('-')[0]
    print(f"\n{'='*60}")
    print(f"Processing: {activity_name}")
    print(f"{'='*60}")

    acc_path = os.path.join(base_dir, activity, "Accelerometer.csv")
    gyro_path = os.path.join(base_dir, activity, "Gyroscope.csv")
    output_path = os.path.join(output_dir, f"{activity_name}_pose_estimation.png")

    try:
        save_pose_estimation_plot(acc_path, gyro_path, output_path, alpha=0.98)
        print(f"✓ Success: {activity_name}")
    except Exception as e:
        print(f"✗ Error processing {activity_name}: {e}")

print(f"\n{'='*60}")
print("All activities processed!")
print(f"{'='*60}\n")
