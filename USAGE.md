# Usage Guide

## Quick Start

```bash
# 1. Setup environment (first time only)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run everything (default - no arguments needed!)
python main.py
```

## Command Reference

### Run All Parts (Default)
```bash
python main.py
```
Executes all three parts on all four activities. Generates all plots and CSV files.

**Note**: Running `python main.py` with no arguments runs everything by default. You can also explicitly use `--all` if you prefer.

### Run Individual Parts

```bash
# Part 1: Activity Classification
python main.py --part1

# Part 2: Step Counting
python main.py --part2

# Part 3: Pose Estimation
python main.py --part3
```

### Run on Specific Activity

```bash
python main.py --activity walking_hand_1
# Runs all parts on just walking_hand_1

python main.py --part2 --activity walking_hand_2
# Runs only Part 2 on walking_hand_2
```

### Skip Plot Generation (Faster)

```bash
python main.py --part2 --no-plots
# Prints results but doesn't save visualizations
```

## What Each Part Does

### Part 1: Activity Classification (`--part1`)

**Input**: Accelerometer data from all activities
**Output**:
- Console: Feature comparison table
- Files: `plots/*_detailed.png`, `plots/feature_comparison.csv`

**What it shows**: Distinguishing features between sitting, standing, and walking

### Part 2: Step Counting (`--part2`)

**Input**: Accelerometer data from walking activities only
**Output**:
- Console: Step count, cadence, method breakdown
- Files: `plots/*_step_analysis.png`

**Methods used**: Peak detection, autocorrelation, zero-crossings

### Part 3: Pose Estimation (`--part3`)

**Input**: Accelerometer + Gyroscope data from all activities
**Output**:
- Console: Pitch/Roll/Yaw statistics
- Files: `plots/*_pose_estimation.png`

**What it estimates**: Device orientation (pitch, roll, yaw angles)

## Example Workflows

### Validate Step Counter
```bash
# Quick validation without plots
python main.py --part2 --no-plots

# Full analysis with visualizations
python main.py --part2
```

### Analyze Single Activity in Detail
```bash
python main.py --activity walking_hand_1
# Generates:
# - walking_hand_1_detailed.png
# - walking_hand_1_step_analysis.png
# - walking_hand_1_pose_estimation.png
```

### Regenerate All Visualizations
```bash
python main.py
# Processes all 4 activities through all 3 parts
# Saves all plots to plots/ directory
```

## Output Files

All files saved to `plots/` directory:

```
plots/
├── feature_comparison.csv              # Part 1: Feature table
├── sitting_hand_detailed.png           # Part 1: Activity analysis
├── standing_hand_detailed.png
├── walking_hand_1_detailed.png
├── walking_hand_2_detailed.png
├── walking_hand_1_step_analysis.png    # Part 2: Step counting
├── walking_hand_2_step_analysis.png
├── sitting_hand_pose_estimation.png    # Part 3: Pose estimation
├── standing_hand_pose_estimation.png
├── walking_hand_1_pose_estimation.png
└── walking_hand_2_pose_estimation.png
```

## Troubleshooting

### "No module named 'xxx'"
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "Activity not found"
```bash
# List available activities
python main.py --help

# Available: sitting_hand, standing_hand, walking_hand_1, walking_hand_2
```

### "FileNotFoundError"
```bash
# Make sure you're in the project root directory
cd "/path/to/HW 1"
python main.py --all
```

## Advanced Usage

### Modify Parameters

To change step counting parameters or pose estimation filter settings, edit the corresponding source files in `src/`:

- `src/stepcount_enhanced.py` - Step counting algorithms
- `src/pose_estimation_improved.py` - Complementary filter settings
- `src/detailed_analysis.py` - Feature extraction

### Add New Activity

1. Place accelerometer/gyroscope CSV files in `data/new_activity/`
2. Edit `main.py` line 33-38 to add activity folder name
3. Run: `python main.py --activity new_activity`

## Getting Help

```bash
python main.py --help
```

For detailed documentation:
- Part 1: `docs/ACTIVITY_ANALYSIS.md`
- Part 2: `docs/STEP_COUNTING_EXPLANATION.md`
- Part 3: `docs/POSE_ESTIMATION_REPORT.md`
