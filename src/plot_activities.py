import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
4
# Folder where all logs are stored
ROOT = "../data"

def read_accel_csv(folder):
    path = os.path.join(ROOT, folder, "Accelerometer.csv")
    df = pd.read_csv(path)
    # Rename to standard order
    df.rename(columns={"x":"accX", "y":"accY", "z":"accZ"}, inplace=True)
    return df

def plot_activity(df, title, filename=None):
    t = df["seconds_elapsed"]
    plt.figure(figsize=(10,4))
    plt.plot(t, df["accX"], label="X")
    plt.plot(t, df["accY"], label="Y")
    plt.plot(t, df["accZ"], label="Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    plt.close()

def compute_features(df):
    mag = np.sqrt(df["accX"]**2 + df["accY"]**2 + df["accZ"]**2)
    return {
        "mean": mag.mean(),
        "std": mag.std(),
        "max": mag.max(),
        "min": mag.min(),
        "peak_count": ((mag.shift(1) < mag) & (mag.shift(-1) < mag)).sum()
    }

folders = [
    "sitting_hand-2025-10-10_14-38-42",
    "standing_hand-2025-10-10_14-38-03",
    "walking_hand_1-2025-10-10_14-39-10",
    "walking_hand_2-2025-10-10_14-39-53"
]

summary = []
output_dir = "../plots"
os.makedirs(output_dir, exist_ok=True)

for f in folders:
    df = read_accel_csv(f)
    features = compute_features(df)
    activity_name = f.split("-")[0]
    features["activity"] = activity_name
    summary.append(features)

    output_file = os.path.join(output_dir, f"{activity_name}.png")
    plot_activity(df, activity_name.replace("_", " ").capitalize(), filename=output_file)

summary_df = pd.DataFrame(summary)
print("\n=== Feature Summary ===")
print(summary_df.round(3))
