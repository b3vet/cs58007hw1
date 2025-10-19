# How to Run the Project

## Simplest Way (Recommended)

```bash
# From project root directory
python main.py
```

That's it! This runs all three parts on all activities by default.

## Command Options

```bash
# Run everything (default - same as no arguments)
python main.py

# Run specific part only
python main.py --part1   # Activity classification
python main.py --part2   # Step counting
python main.py --part3   # Pose estimation

# Run on specific activity
python main.py --activity walking_hand_1

# Skip plot generation (faster)
python main.py --no-plots

# Combine options
python main.py --part2 --activity walking_hand_1 --no-plots
```

## First Time Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py
```

## What Gets Executed

When you run `python main.py` with no arguments:

1. **Part 1**: Activity Classification & Feature Analysis
   - Processes all 4 activities (sitting, standing, walking x2)
   - Generates detailed plots showing features
   - Creates feature comparison CSV

2. **Part 2**: Step Counting
   - Processes walking activities only
   - Uses 3 methods: peak detection, autocorrelation, zero-crossings
   - Generates step analysis visualizations

3. **Part 3**: Pose Estimation
   - Processes all 4 activities
   - Estimates pitch, roll, yaw angles
   - Generates orientation plots

## Output

All results saved to `plots/` directory:
- Feature analysis plots
- Step counting visualizations  
- Pose estimation graphs
- CSV file with feature comparison

## Need Help?

```bash
python main.py --help
```

## More Documentation

- [USAGE.md](../USAGE.md) - Detailed usage guide
- [README.md](../README.md) - Project overview
- [ACTIVITY_ANALYSIS.md](ACTIVITY_ANALYSIS.md) - Part 1 details
- [STEP_COUNTING_EXPLANATION.md](STEP_COUNTING_EXPLANATION.md) - Part 2 details
- [POSE_ESTIMATION_REPORT.md](POSE_ESTIMATION_REPORT.md) - Part 3 details
