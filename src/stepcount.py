import csv
import math
from typing import List, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting is optional

# ---------- CSV READERS (no pandas) ----------
def read_iphone_accel_csv(csv_path: str):
    """
    Reads Sensor Logger style Accelerometer.csv with columns:
    time, seconds_elapsed, z, y, x  (order may vary, but names match)
    Returns dict with 't' (seconds) and axes 'x','y','z' as numpy arrays.
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [h.strip().lower() for h in header]

        # figure out indices
        def idx(name):
            return cols.index(name)

        if "seconds_elapsed" in cols:
            t_idx = idx("seconds_elapsed")
        elif "time" in cols:
            t_idx = idx("time")  # if needed, we will normalize later
        else:
            raise ValueError("CSV must have 'seconds_elapsed' or 'time' column.")

        # axis columns are named 'x','y','z'
        x_idx = idx("x")
        y_idx = idx("y")
        z_idx = idx("z")

        t, x, y, z = [], [], [], []
        for row in reader:
            try:
                t.append(float(row[t_idx]))
                x.append(float(row[x_idx]))
                y.append(float(row[y_idx]))
                z.append(float(row[z_idx]))
            except:
                # skip malformed lines
                continue

        t = np.array(t, dtype=np.float64)
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        z = np.array(z, dtype=np.float64)

        # if 'time' not normalized, make it start at 0 and convert to seconds if looks like ms
        if "seconds_elapsed" not in cols:
            t = t - t[0]
            # heuristic: if large numbers, assume ms
            if np.median(np.diff(t)) > 10.0:
                t = t / 1000.0

        return {"t": t, "x": x, "y": y, "z": z}

# ---------- SIGNAL UTILITIES (hand-rolled) ----------
def estimate_fs(t: np.ndarray) -> float:
    diffs = np.diff(t)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    return 1.0 / float(np.median(diffs)) if len(diffs) else 50.0

def single_pole_lowpass_coeff(fc: float, fs: float) -> float:
    """
    Returns alpha for y[n] = alpha*y[n-1] + (1-alpha)*x[n]
    where alpha = exp(-2*pi*fc/fs) to approximate analog -3dB at fc.
    """
    if fc <= 0.0 or fs <= 0.0:
        return 0.0
    return math.exp(-2.0 * math.pi * fc / fs)

def lowpass_iir_1st(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    a = single_pole_lowpass_coeff(fc, fs)
    y = np.zeros_like(x)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = a * y[n - 1] + (1.0 - a) * x[n]
    return y

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    y = np.zeros_like(x)
    s = 0.0
    q = [0.0] * k
    qi = 0
    for i, v in enumerate(x):
        s -= q[qi]
        q[qi] = v
        s += v
        qi = (qi + 1) % k
        if i < k - 1:
            y[i] = s / (i + 1)
        else:
            y[i] = s / k
    return y

def magnitude(x, y, z):
    return np.sqrt(x * x + y * y + z * z)

# ---------- PEAK DETECTOR (hand-rolled) ----------
def local_maxima_indices(x: np.ndarray, min_distance_samples: int, threshold: float) -> List[int]:
    """
    Naive peak finder:
      - peak must be strictly greater than immediate neighbors
      - x[i] >= threshold
      - enforce min distance between accepted peaks
    """
    peaks = []
    last_i = -10**9
    N = len(x)
    for i in range(1, N - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] >= threshold:
            if i - last_i >= min_distance_samples:
                peaks.append(i)
                last_i = i
    return peaks

# ---------- SLIDING WINDOW ----------
def sliding_windows(N: int, win: int, step: int):
    i = 0
    while i + win <= N:
        yield i, i + win
        i += step

# ---------- MAIN PIPELINE ----------
def count_steps_from_csv(csv_path: str,
                         gravity_fc: float = 0.3,
                         smoothing_ma: int = 5,
                         window_sec: float = 1.5,
                         overlap: float = 0.5,
                         min_step_hz: float = 0.75,
                         max_step_hz: float = 3.0,
                         debug_plot: bool = False) -> Dict:
    data = read_iphone_accel_csv(csv_path)
    t, ax, ay, az = data["t"], data["x"], data["y"], data["z"]
    N = len(t)
    if N < 10:
        return {"fs": 0.0, "steps": 0, "duration_sec": 0.0, "cadence_spm": 0.0}

    fs = estimate_fs(t)
    duration = float(t[-1] - t[0])

    # 1) Estimate gravity on each axis with low-pass; subtract to get linear acceleration
    gx = lowpass_iir_1st(ax, gravity_fc, fs)
    gy = lowpass_iir_1st(ay, gravity_fc, fs)
    gz = lowpass_iir_1st(az, gravity_fc, fs)
    lx = ax - gx
    ly = ay - gy
    lz = az - gz

    # 2) Magnitude of linear acceleration
    mag = magnitude(lx, ly, lz)

    # 3) Smooth the magnitude (moving average, hand-rolled)
    mag_s = moving_average(mag, smoothing_ma)

    # 4) Sliding-window adaptive threshold
    win = max(4, int(window_sec * fs))
    step = max(1, int(win * (1.0 - overlap)))
    global_peaks = []

    # min/max step periods -> min/max spacing in samples
    min_dist = max(1, int(fs / max_step_hz))  # closest peaks
    max_dist = max(1, int(fs / min_step_hz))  # not enforced here, informative

    for s, e in sliding_windows(len(mag_s), win, step):
        w = mag_s[s:e]
        if len(w) < 3:
            continue
        w_mean = float(np.mean(w))
        w_std = float(np.std(w))

        # Adaptive threshold = mean + k*std, k chosen conservatively
        thr = w_mean + 0.25 * w_std

        # Find local peaks above threshold; shift indices by window start
        peaks = local_maxima_indices(w, min_distance_samples=min_dist, threshold=thr)
        peaks = [p + s for p in peaks]

        global_peaks.extend(peaks)

    # Deduplicate peaks that may appear in overlapping windows
    # Keep peaks separated by at least min_dist; keep the stronger one.
    if global_peaks:
        global_peaks = sorted(set(global_peaks))
        dedup = []
        last = None
        for idx in global_peaks:
            if last is None or idx - last >= min_dist:
                dedup.append(idx)
                last = idx
            else:
                # if too close, keep the larger magnitude
                if mag_s[idx] > mag_s[last]:
                    dedup[-1] = idx
                    last = idx
        global_peaks = dedup

    steps = len(global_peaks)
    cadence_spm = (steps / duration) * 60.0 if duration > 0 else 0.0

    if debug_plot and plt is not None:
        tt = t - t[0]
        plt.figure(figsize=(10,4))
        plt.plot(tt, mag_s, label="|lin acc| (smoothed)")
        plt.scatter(tt[global_peaks], mag_s[global_peaks], marker="o", s=24, label="peaks")
        plt.title("Step Count Debug")
        plt.xlabel("Time (s)")
        plt.ylabel("m/s²")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "fs": fs,
        "steps": steps,
        "duration_sec": duration,
        "cadence_spm": cadence_spm
    }
