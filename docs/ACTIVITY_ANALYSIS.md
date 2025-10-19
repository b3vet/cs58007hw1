# Detailed Activity Classification Analysis
## Visual Inspection and Feature Analysis of Accelerometer Data

---

## Overview
This document provides a detailed analysis of accelerometer data collected during four activities:
- **Sitting** (with phone in hand)
- **Standing** (with phone in hand)
- **Walking** (with phone in hand) - 2 trials

---

## Key Distinguishing Features

### 1. **Acceleration Magnitude (Most Important Feature)**

#### Visual Observations:
- **Sitting**: Magnitude plot shows relatively flat signal oscillating around **0.167 m/s²**
  - Range: 0.007 - 0.578 m/s²
  - Very tight distribution (narrow histogram)

- **Standing**: Magnitude slightly higher than sitting at **0.231 m/s²**
  - Range: 0.013 - 0.783 m/s²
  - Slightly more variability than sitting due to natural body sway

- **Walking**: Dramatically different! Magnitude around **0.99-1.02 m/s²**
  - Range: 0.014 - 3.194 m/s² (much wider)
  - **6x higher mean than sitting**
  - **4-5x higher mean than standing**

#### Why This Works:
Walking involves periodic arm swinging and body movement, generating much larger acceleration changes. Static activities (sitting/standing) only capture hand tremors and natural micro-movements.

---

### 2. **Variability (Standard Deviation)**

#### Visual Observations:
- **Sitting**: Std = 0.094 m/s² (very stable)
  - Magnitude plot shows consistent low-amplitude fluctuations
  - Histogram is narrow and peaked around the mean

- **Standing**: Std = 0.129 m/s² (slightly more variable)
  - More outliers visible in the magnitude plot
  - Natural postural adjustments cause small variations

- **Walking**: Std = 0.447 m/s² (**~5x higher than sitting**)
  - Magnitude plot shows large, rhythmic oscillations
  - Histogram is wide and more spread out
  - Reflects the periodic acceleration/deceleration of walking motion

#### Why This Works:
Walking is inherently periodic with alternating acceleration patterns (step cycle). Static activities have minimal variation, only capturing sensor noise and minor hand movements.

---

### 3. **Frequency Domain Analysis (FFT - Critical for Walking Detection)**

#### Visual Observations:
- **Sitting**:
  - FFT shows no clear dominant frequency
  - Power spectrum is scattered across many frequencies
  - Peak at 1.11 Hz but very weak (power ~0.025)
  - **Interpretation**: Random, non-periodic motion

- **Standing**:
  - Similar to sitting - no strong dominant frequency
  - Peak at 0.17 Hz (very low frequency, likely postural drift)
  - Power spectrum is noisy
  - **Interpretation**: Random micro-movements, no rhythm

- **Walking**:
  - **Very clear dominant frequency at ~1.50 Hz** (both trials!)
  - Strong power peak (0.175-0.180) - **7x stronger than sitting**
  - This frequency corresponds to **step rate** (~1.5 steps per second = 90 steps/minute)
  - **Interpretation**: Clear periodic pattern characteristic of gait

#### Why This Works:
Human walking has a natural cadence (step frequency) typically between 1-2 Hz. FFT captures this periodicity perfectly, making it a highly discriminative feature. Static activities have no such periodicity.

---

### 4. **Temporal Patterns (Time-Series Visual Inspection)**

#### Sitting:
- Raw accelerometer plot shows:
  - Random, irregular fluctuations
  - No visible repeating patterns
  - All three axes (X, Y, Z) oscillate around relatively constant values
  - Z-axis dominates (negative values ~-0.09 m/s², likely gravity component)

#### Standing:
- Similar to sitting but with:
  - Occasional larger spikes (postural corrections)
  - Slightly more variation in amplitude
  - Still no periodic structure
  - Moving statistics plot shows gradual drift over time (natural body sway)

#### Walking:
- **Dramatically different!**
  - Clear, regular oscillations visible across all time periods
  - Z-axis shows pronounced periodic swings (±2 m/s² range)
  - X and Y axes also show synchronized periodic patterns
  - Moving statistics plot reveals consistent oscillation amplitude throughout
  - **Visual pattern repeats every ~0.67 seconds** (matches 1.5 Hz frequency)

#### Why This Works:
Walking creates a characteristic "sawtooth" or sinusoidal pattern due to the step cycle: acceleration during leg push-off, deceleration during swing phase. This is immediately visible in time-series plots.

---

### 5. **Axis-Specific Characteristics**

#### Distribution Comparison (Box Plots):

**Sitting:**
- X: Centered near 0, small range (-0.4 to +0.4 m/s²)
- Y: Very narrow, centered near 0
- Z: Most variable, centered around -0.09 m/s² (gravity)

**Standing:**
- Similar to sitting but with more outliers
- Z-axis has many outliers showing postural adjustments

**Walking:**
- **All axes have much wider distributions**
- X: Range approximately -1 to +1 m/s²
- Y: Range approximately -1.5 to +1.5 m/s²
- Z: **Huge range -2.5 to +1.5 m/s²** (dominant axis for walking)
- Many outliers indicating dynamic movement

#### Why This Works:
Walking engages all three spatial dimensions. The Z-axis (vertical) shows the largest changes due to the up-down motion of walking. Static activities show minimal axis variation.

---

### 6. **Signal Energy**

#### Quantitative Comparison:
- **Sitting**: Total energy = 0.037 m²/s⁴
- **Standing**: Total energy = 0.070 m²/s⁴ (~2x sitting)
- **Walking**: Total energy = 1.17-1.24 m²/s⁴ (**~33x sitting!**)

#### Visual Confirmation:
- Magnitude plots show walking has much larger area under the curve
- Spectral energy (FFT domain) is also **much higher for walking**

#### Why This Works:
Energy captures overall motion intensity. Walking requires significant muscular activity and generates large accelerations, resulting in orders of magnitude more signal energy.

---

### 7. **Peak Detection and Periodicity**

#### Observations:
- **Sitting**: 127 peaks detected
  - But these are random, irregular peaks
  - No consistent spacing

- **Standing**: 129 peaks
  - Similar to sitting - random peaks

- **Walking Trial 1**: 166 peaks over 21.4 seconds
  - **~7.8 peaks/second** average
  - More importantly: peaks are **regularly spaced**
  - Visual inspection confirms periodic peaks matching step frequency

- **Walking Trial 2**: 154 peaks over 21.4 seconds
  - Consistent with Trial 1

#### Why This Works:
Each walking step creates a characteristic acceleration peak. The regularity and spacing of these peaks distinguish walking from random motion in static activities.

---

### 8. **Magnitude Distribution Shape**

#### Histogram Analysis:

**Sitting:**
- Narrow, bell-shaped distribution
- Strongly centered around mean (0.167)
- Most values within ±1 standard deviation
- Right-skewed tail (occasional larger movements)

**Standing:**
- Similar shape to sitting but slightly wider
- Still predominantly centered
- More pronounced right tail (postural adjustments)

**Walking:**
- **Much wider, more uniform distribution**
- Bimodal tendency (two slight humps visible)
- Extends from ~0.2 to ~2.5 m/s²
- This reflects the oscillation between low (mid-stride) and high (foot strike) acceleration

#### Why This Works:
Walking's periodic nature creates a more spread-out distribution as the acceleration cycles through high and low values. Static activities cluster tightly around a constant low value.

---

### 9. **Moving Statistics (Rolling Mean and Std)**

#### Visual Patterns:

**Sitting:**
- Rolling mean is relatively flat (~0.15-0.20 m/s²)
- Rolling std band is narrow and consistent
- Small variations reflect random fluctuations

**Standing:**
- Rolling mean shows more variation than sitting
- Slight increasing/decreasing trends (postural drift)
- Rolling std band slightly wider

**Walking:**
- Rolling mean oscillates but maintains higher level (~1.0 m/s²)
- **Rolling std band is much wider** (0.25-0.75 m/s²)
- Shows the signal is consistently variable throughout
- Indicates sustained dynamic activity

#### Why This Works:
Moving statistics reveal the temporal consistency of an activity. Walking maintains high variability throughout, while static activities remain consistently low in both mean and variance.

---

## Summary: Best Features for Activity Classification

### **Tier 1 (Highest Discriminative Power):**

1. **Magnitude Mean** - Single best feature
   - Clear separation: Sitting (~0.17) < Standing (~0.23) << Walking (~1.0)
   - Simple threshold can classify with high accuracy

2. **Magnitude Standard Deviation** - Excellent separator
   - Static (~0.09-0.13) vs. Dynamic (~0.45)
   - Captures variability difference

3. **Dominant Frequency** - Perfect for detecting walking
   - Static: No clear frequency
   - Walking: Strong peak at ~1.5 Hz
   - Can detect any periodic activity

### **Tier 2 (Strong Supporting Features):**

4. **Total Signal Energy** - Clear separator
   - 33x difference between sitting and walking
   - Robust to orientation changes

5. **Magnitude Range (Max - Min)**
   - Sitting: 0.57, Standing: 0.77, Walking: 2.61-3.18
   - Simple but effective

6. **Z-Axis Standard Deviation**
   - Walking: 0.87-0.91 vs. Sitting: 0.14
   - Captures vertical motion component

### **Tier 3 (Useful for Edge Cases):**

7. **Spectral Power at Dominant Frequency**
   - Distinguishes periodic from random motion

8. **Peak Count Rate**
   - Walking has more peaks per second

9. **Axis Correlations**
   - Walking shows different correlation patterns
   - Can help distinguish orientation

---

## Distinguishing Between Sitting and Standing

These are harder to separate but possible using:

1. **Magnitude Mean**: Standing (0.231) > Sitting (0.167)
   - Standing requires muscle engagement for postural control

2. **Standard Deviation**: Standing (0.129) > Sitting (0.094)
   - Standing has more micro-movements (body sway)

3. **Maximum Acceleration**: Standing (0.783) > Sitting (0.578)
   - Postural adjustments create larger spikes

4. **Number of Outliers**: Standing has more (visible in box plots)
   - More frequent corrective movements

---

## Practical Classification Strategy

### Simple Rule-Based Approach:
```python
if magnitude_mean > 0.5:
    return "Walking"
elif dominant_frequency > 1.0 and dominant_power > 0.1:
    return "Walking"  # Backup check
elif magnitude_std > 0.3:
    return "Walking"  # Another backup
elif magnitude_mean > 0.2:
    return "Standing"
else:
    return "Sitting"
```

### Machine Learning Approach:
Use all Tier 1 and Tier 2 features (9 total) with:
- Decision Tree / Random Forest (works great with these features)
- SVM with RBF kernel
- Simple Neural Network

Expected accuracy: >95% for walking vs. static, ~80-90% for sitting vs. standing

---

## Conclusion

The accelerometer data shows **very clear distinctions** between walking and static activities (sitting/standing). Walking is characterized by:
- High magnitude (~6x higher)
- High variability (~5x higher)
- Clear periodicity (~1.5 Hz)
- High energy (~33x higher)

Sitting vs. standing is more subtle but distinguishable through:
- Magnitude differences (~38% higher for standing)
- Variability differences (~37% higher for standing)
- More frequent peaks/outliers in standing

These features provide robust, interpretable, and effective classification capabilities for human activity recognition.
