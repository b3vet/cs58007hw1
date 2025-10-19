# Pose Estimation Code Analysis

## Current Implementation Review

### What the Code Does:
The current implementation uses a **Complementary Filter** to estimate pitch and roll angles from accelerometer and gyroscope data.

### Algorithm Overview:
```
1. Read accelerometer and gyroscope data
2. Interpolate gyroscope data to match accelerometer timestamps
3. For each time step:
   - Calculate pitch/roll from accelerometer (gravity-based)
   - Integrate gyroscope data to get pitch/roll change
   - Combine using weighted average: α × gyro + (1-α) × accel
```

---

## Issues Found

### ✗ Issue 1: Incorrect Pitch Calculation from Accelerometer
**Current code (line 34):**
```python
pitch_acc = math.degrees(math.atan2(ay[i], math.sqrt(ax[i]**2 + az[i]**2)))
```

**Problem:** This formula is INCORRECT for pitch calculation.

**Correct formula:**
```python
pitch_acc = math.degrees(math.atan2(-ax[i], math.sqrt(ay[i]**2 + az[i]**2)))
```

**Why?** Pitch is rotation around Y-axis. When phone tilts forward/backward (pitch), the X-axis acceleration changes, not Y-axis.

---

### ✗ Issue 2: Incorrect Roll Calculation from Accelerometer
**Current code (line 35):**
```python
roll_acc = math.degrees(math.atan2(-ax[i], math.sqrt(ay[i]**2 + az[i]**2)))
```

**Problem:** This is actually calculating pitch, not roll!

**Correct formula:**
```python
roll_acc = math.degrees(math.atan2(ay[i], math.sqrt(ax[i]**2 + az[i]**2)))
```

**Why?** Roll is rotation around X-axis. When phone tilts left/right (roll), the Y-axis acceleration changes.

---

### ✗ Issue 3: Swapped Variables
The pitch and roll formulas are literally swapped! This will cause:
- Reported "pitch" is actually "roll"
- Reported "roll" is actually "pitch"

---

### ✗ Issue 4: Missing Gyroscope Bias Correction
Gyroscopes have a small constant bias (drift) that accumulates over time.

**Impact:** Without bias correction, the estimated angles will drift over time, even when the device is stationary.

**Solution:** Estimate and subtract the bias (mean value during stationary period).

---

### ✗ Issue 5: No Yaw Estimation
The current code only estimates pitch and roll. **Yaw (rotation around Z-axis) cannot be estimated from accelerometer alone** because gravity doesn't change with yaw rotation.

**Options:**
1. Accept that yaw is not available (requires magnetometer)
2. Integrate gyroscope Z-axis for relative yaw changes
3. Note the limitation in documentation

---

### ✗ Issue 6: Fixed dt Calculation
**Current code (lines 27-28):**
```python
fs = 1/np.median(np.diff(t))
dt = 1/fs
```

**Problem:** Uses fixed dt for entire recording, but actual timestamps may vary.

**Better approach:** Use actual time difference for each step:
```python
dt = t[i] - t[i-1]  # Actual time step
```

---

### ✓ Good Points

1. **Complementary filter** is the right choice - simple and effective
2. **Data interpolation** correctly handles different sampling rates
3. **Alpha = 0.98** is reasonable (trusts gyro more for short-term, accel for long-term)
4. **Code structure** is clean and readable

---

## Recommended Improvements

### Priority 1 (Critical - Correctness):
1. ✅ Fix pitch/roll formulas (swap them)
2. ✅ Use variable dt based on actual timestamps
3. ✅ Add gyroscope bias correction

### Priority 2 (Important - Robustness):
4. ✅ Add noise filtering for accelerometer
5. ✅ Handle edge cases (division by zero, invalid data)
6. ✅ Add validation checks

### Priority 3 (Nice to have - Features):
7. ✅ Add quaternion-based orientation (more accurate)
8. ✅ Add yaw estimation (with drift warning)
9. ✅ Add comprehensive visualization
10. ✅ Add performance metrics

---

## Mathematical Background

### Accelerometer-based Orientation:

When device is stationary, accelerometer measures gravity vector:
- **Pitch** (rotation around Y-axis): `atan2(-ax, sqrt(ay² + az²))`
- **Roll** (rotation around X-axis): `atan2(ay, sqrt(ax² + az²))`
- **Yaw**: Cannot be determined from gravity alone

### Gyroscope Integration:

Gyroscope measures angular velocity (rad/s):
```
angle(t) = angle(t-1) + gyro × dt
```

### Complementary Filter:

Combines both sources:
```
angle = α × angle_gyro + (1-α) × angle_accel
```

Where:
- α ≈ 0.98: Trust gyro for short-term (no noise)
- (1-α) ≈ 0.02: Trust accel for long-term (no drift)

---

## Axis Convention (iPhone Sensor Logger)

Standard smartphone axes:
```
     Y (points up when phone upright)
     |
     |
     +---- X (points right)
    /
   /
  Z (points out of screen)
```

- **Pitch**: Rotation around Y-axis (tilt forward/backward)
- **Roll**: Rotation around X-axis (tilt left/right)
- **Yaw**: Rotation around Z-axis (rotate on table)

---

## Conclusion

The current implementation has the **RIGHT IDEA** (complementary filter) but **WRONG FORMULAS** (pitch/roll swapped).

After fixing the formulas and adding improvements, the code will work correctly.
