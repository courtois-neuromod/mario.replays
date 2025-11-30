# mario.replays

Extract game data from Super Mario Bros `.bk2` replay files. Processes recordings to generate frame-by-frame game variables, summary statistics, and optional video files.

## What You Get

For each `.bk2` replay file:

1. **JSON sidecar** - Summary statistics:
   - Level cleared? Score gained, distance traveled
   - Enemies killed, powerups/coins collected
   - Lives lost, hits taken, bricks destroyed

2. **Game variables** - Frame-by-frame data:
   - All RAM variables (player position, score, lives, etc.)
   - Button states for each frame
   - Structured as JSON dictionary

3. **Video replay** (optional) - MP4 with audio
4. **RAM dumps** (optional) - Complete memory state per frame

## Quick Start

```bash
# Install
git clone git@github.com:courtois-neuromod/mario.replays
cd mario.replays
pip install -r requirements.txt
pip install -e .

# Or with airoh
pip install airoh
invoke setup-env

# Process replays
invoke create-replays
```

## Usage

### With Airoh (Recommended)

```bash
# Process with default settings
invoke create-replays

# Include videos and use parallel processing
invoke create-replays --save-videos --n-jobs 8

# Custom paths
invoke create-replays \
  --datapath /data/mario \
  --output /data/replays
```

### Direct Python Script

```bash
python code/mario_replays/create_replays/create_replays.py \
  --datapath sourcedata/mario \
  --output outputdata/replays \
  --save_variables \
  --n_jobs -1
```

## Requirements

- Python ≥ 3.8
- Mario dataset with `.bk2` replay files and stimuli folder
- stable-retro

## Configuration

Edit `invoke.yaml`:

```yaml
mario_dataset: sourcedata/mario
output_dir: outputdata/replays
n_jobs: -1              # Use all CPU cores
save_videos: false
save_variables: true
```

## Output Structure

```
outputdata/replays/
└── sub-XX/
    └── ses-XXX/
        └── beh/
            ├── infos/       # JSON summary files
            ├── variables/   # Game variables
            ├── videos/      # MP4 replays (optional)
            └── ramdumps/    # RAM states (optional)
```

## Available Tasks

```bash
invoke --list                    # View all tasks
invoke create-replays           # Process replay files
invoke setup-env                # Install dependencies
invoke setup-mario-dataset      # Download dataset via Datalad
```

### Task Options

- `--datapath` - Mario dataset location
- `--output` - Output directory
- `--n-jobs` - Parallel jobs (-1 = all cores)
- `--save-videos` - Generate MP4 files
- `--save-variables` - Save game variables (default: true)
- `--save-ramdumps` - Save RAM dumps
- `--verbose` - Detailed logging

## Data Format

### Summary JSON

```json
{
  "Subject": "sub-01",
  "World": "1",
  "Level": "1",
  "Duration": 45.3,
  "Cleared": true,
  "ScoreGained": 2450,
  "Enemies_killed": 8,
  "Powerups_collected": 2,
  "CoinsGained": 12
}
```

### Variables JSON

```json
{
  "filename": "sub-01_ses-001_..._rep-00.bk2",
  "level": "w1l1",
  "score": [0, 0, 5, 5, ...],
  "lives": [2, 2, 2, ...],
  "player_x_posLo": [96, 97, ...],
  "A": [0, 0, 1, 0, ...],
  "B": [0, 1, 1, ...]
}
```

## Troubleshooting

**"No retro integration"**: Ensure stimuli folder exists in mario dataset with ROM and integration files

**Slow processing**: Increase `--n-jobs`, disable `--save-videos`, or process specific subjects only

**Memory issues**: Reduce `n_jobs`, disable `--save-ramdumps`

## Related Projects

- [mario](https://github.com/courtois-neuromod/mario) - Main dataset
- [mario.annotations](https://github.com/courtois-neuromod/mario.annotations) - Event annotations
- [mario.scenes](https://github.com/courtois-neuromod/mario.scenes) - Scene analysis

Part of the [Courtois NeuroMod](https://www.cneuromod.ca/) project.
