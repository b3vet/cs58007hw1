# Human Activity Recognition and Pose Estimation

Complete implementation of accelerometer and gyroscope data analysis for activity recognition, step counting, and pose estimation.

## Quick Start

```bash
# Setup (first time only)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run everything (default)
python main.py
```

That's it! Running `python main.py` with no arguments will execute all three parts on all activities.

## Usage

```bash
# Run all parts (default)
python main.py

# Run specific part only
python main.py --part1   # Activity analysis
python main.py --part2   # Step counting  
python main.py --part3   # Pose estimation

# Run on specific activity
python main.py --activity walking_hand_1

# Skip plot generation (faster)
python main.py --no-plots
```

## Project Status

✅ **Part 1: Activity Classification** - COMPLETE  
✅ **Part 2: Step Counting** - COMPLETE  
✅ **Part 3: Pose Estimation** - COMPLETE (bugs fixed)

## Key Results

| Part | Result | Status |
|------|--------|--------|
| **Part 1** | Walking: 6x higher magnitude, 1.5 Hz periodicity | ✅ Clear separation |
| **Part 2** | 35-41 steps detected (expected 30-45) | ✅ Accurate |
| **Part 3** | Pitch/Roll: ±20° (original had formulas swapped!) | ✅ Fixed |

## Project Structure

```
HW 1/
├── main.py                 ⭐ Run this! (python main.py)
├── src/                    Implementation modules
├── data/                   Sensor recordings
├── plots/                  Generated outputs
├── docs/                   Documentation
└── README.md               This file
```

## Output Example

```
$ python main.py --no-plots

================================================================================
                  HUMAN ACTIVITY RECOGNITION & POSE ESTIMATION
================================================================================

Timestamp: 2025-10-19 20:44:51
Activities to process: 4
Parts to run: Part 1, Part 2, Part 3
Save plots: False

... (comprehensive logs for each part) ...

✓ All requested parts executed successfully
```

## Documentation

- [USAGE.md](USAGE.md) - Detailed usage guide
- [docs/ACTIVITY_ANALYSIS.md](docs/ACTIVITY_ANALYSIS.md) - Part 1 technical details
- [docs/STEP_COUNTING_EXPLANATION.md](docs/STEP_COUNTING_EXPLANATION.md) - Part 2 technical details  
- [docs/POSE_ESTIMATION_REPORT.md](docs/POSE_ESTIMATION_REPORT.md) - Part 3 bug fixes

## Quick Reference

| Command | Description |
|---------|-------------|
| `python main.py` | Run all parts (default) |
| `python main.py --part1` | Activity classification only |
| `python main.py --part2` | Step counting only |
| `python main.py --part3` | Pose estimation only |
| `python main.py --activity walking_hand_1` | Single activity |
| `python main.py --no-plots` | Skip visualizations |
| `python main.py --help` | Show all options |

## All Complete! 🎉

Just run `python main.py` and everything works!
