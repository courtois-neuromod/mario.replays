"""
Invoke tasks for mario.replays project using airoh.

This module provides tasks for processing Mario dataset replay files
and extracting game variables, frames, and metadata.
"""

from invoke import task
import os
import os.path as op

# Import airoh utility tasks
from airoh.utils import setup_env_python, ensure_dir_exist
from airoh.datalad import get_data

BASE_DIR = op.dirname(op.abspath(__file__))


@task
def create_replays(
    c,
    datapath=None,
    stimuli=None,
    output=None,
    n_jobs=None,
    save_videos=None,
    save_variables=None,
    save_ramdumps=None,
    save_confs=None,
    simple=False,
    verbose=False,
    subjects=None,
    sessions=None,
):
    """
    Process Mario dataset replay files and extract game data.

    This task processes .bk2 replay files from the Mario dataset and generates:
    - JSON sidecar files with game metadata
    - Game variable files (optional)
    - Playback videos (optional)
    - RAM dumps (optional)

    Parameters
    ----------
    c : invoke.Context
        The Invoke context (automatically provided).
    datapath : str, optional
        Path to the mario dataset root. Defaults to sourcedata/mario from invoke.yaml.
    stimuli : str, optional
        Path to stimuli files. Defaults to stimuli_path from invoke.yaml.
    output : str, optional
        Output directory for processed files. Defaults to output_dir from invoke.yaml.
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores). Defaults to n_jobs from invoke.yaml.
    save_videos : bool, optional
        Save playback videos (.mp4). Defaults to save_videos from invoke.yaml.
    save_variables : bool, optional
        Save game variables (.json). Defaults to save_variables from invoke.yaml.
    save_ramdumps : bool, optional
        Save RAM dumps (.npz). Defaults to save_ramdumps from invoke.yaml.
    save_confs : bool, optional
        Save psychophysical confounds (.npy). Defaults to save_confs from invoke.yaml.
    simple : bool, optional
        Use simplified game version. Default: False.
    verbose : bool, optional
        Enable verbose output. Default: False.
    subjects : str, optional
        Space-separated subject IDs to process (e.g., "sub-01 sub-02").
        If None, processes all subjects.
    sessions : str, optional
        Space-separated session IDs to process (e.g., "ses-001 ses-002").
        If None, processes all sessions.

    Examples
    --------
    Process replays with default settings:
    ```bash
    invoke create-replays
    ```

    Process with videos and verbose output:
    ```bash
    invoke create-replays --save-videos --verbose
    ```

    Use custom paths and parallel processing:
    ```bash
    invoke create-replays \
      --datapath /data/mario \
      --output /data/derivatives/replays \
      --n-jobs 8
    ```

    Process specific subjects and sessions:
    ```bash
    invoke create-replays --subjects "sub-01 sub-02" --sessions "ses-001"
    ```
    """
    # Resolve paths from configuration or arguments
    if datapath is None:
        datapath = c.config.get("mario_dataset", "sourcedata/mario")

    if stimuli is None:
        stimuli = c.config.get("stimuli_path", op.join(datapath, "stimuli"))

    if output is None:
        output = c.config.get("output_dir", "outputdata/replays")

    if n_jobs is None:
        n_jobs = c.config.get("n_jobs", -1)

    # Resolve boolean flags from config if not explicitly set via CLI
    if save_videos is None:
        save_videos = c.config.get("save_videos", False)

    if save_variables is None:
        save_variables = c.config.get("save_variables", True)

    if save_ramdumps is None:
        save_ramdumps = c.config.get("save_ramdumps", False)

    if save_confs is None:
        save_confs = c.config.get("save_confs", False)

    # Validate paths
    if not op.exists(datapath):
        raise FileNotFoundError(
            f"❌ Mario dataset not found at: {datapath}\n"
            "   Run 'invoke setup-mario-dataset' or specify --datapath"
        )

    # Build command
    cmd = [
        "python",
        "code/mario_replays/create_replays/create_replays.py",
        "--datapath", datapath,
        "--output", output,
        "--n_jobs", str(n_jobs),
    ]

    if stimuli:
        cmd.extend(["--stimuli", stimuli])

    if save_videos:
        cmd.append("--save_videos")

    if save_variables:
        cmd.append("--save_variables")

    if save_ramdumps:
        cmd.append("--save_ramdumps")

    if save_confs:
        cmd.append("--save_confs")

    if simple:
        cmd.append("--simple")

    if verbose:
        cmd.append("--verbose")

    if subjects:
        cmd.extend(["--subjects", subjects])

    if sessions:
        cmd.extend(["--sessions", sessions])

    # Display execution info
    print("🎮 Processing Mario replays...")
    print(f"   Dataset: {datapath}")
    print(f"   Output: {output}")
    print(f"   Parallel jobs: {n_jobs}")
    print(f"   Save videos: {save_videos}")
    print(f"   Save variables: {save_variables}")
    if subjects:
        print(f"   Subjects: {subjects}")
    if sessions:
        print(f"   Sessions: {sessions}")
    print()

    # Run the processing script
    c.run(" ".join(cmd), pty=True)

    print("✅ Replay processing complete!")


@task
def setup_mario_dataset(c, use_datalad=True):
    """
    Set up the Mario dataset with replay files and stimuli.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    use_datalad : bool, optional
        Use datalad to install the dataset. Default: True.

    Examples
    --------
    ```bash
    invoke setup-mario-dataset
    ```
    """
    if use_datalad:
        print("📦 Setting up Mario dataset with Datalad...")
        command = (
            "mkdir -p sourcedata && "
            "cd sourcedata && "
            "datalad install git@github.com:courtois-neuromod/mario && "
            "cd mario && "
            "datalad get */*/*/*.bk2 && "
            "datalad get */*/*/*.tsv && "
            "rm -rf stimuli && "
            "datalad install git@github.com:courtois-neuromod/mario.stimuli && "
            "mv mario.stimuli stimuli && "
            "cd stimuli && "
            "datalad get ."
        )
        c.run(command, pty=True)
        print("✅ Mario dataset setup complete!")
    else:
        print("⚠️  Please manually download the Mario dataset and place it in sourcedata/mario")


@task
def setup_env(c, compute_canada=False):
    """
    Set up the Python environment for mario.replays.

    Parameters
    ----------
    c : invoke.Context
        The Invoke context.
    compute_canada : bool, optional
        Use Compute Canada-specific setup (builds stable-retro from source).
        Default: False.

    Examples
    --------
    Standard setup:
    ```bash
    invoke setup-env
    ```

    Compute Canada setup:
    ```bash
    invoke setup-env --compute-canada
    ```
    """
    print("🐍 Setting up mario.replays environment...")
    print("📦 Installing required packages...")

    env_setup_lines = [
        "set -e",
        "python -m venv env",
        "source env/bin/activate",
        "which python",
        "pip install -e .",
    ]

    if compute_canada:
        print("📦 Building stable-retro from source for Compute Canada...")
        env_setup_lines.extend(
            [
                "git clone https://github.com/FaramaFoundation/stable-retro.git || true",
                "pip install -e stable-retro",
            ]
        )

    env_setup_lines.append("deactivate")

    c.run("\n".join(env_setup_lines), pty=True)

    print("✅ Environment setup complete!")


@task
def full_pipeline(c):
    """
    Run the full processing pipeline: setup environment, get data, process replays.

    Examples
    --------
    ```bash
    invoke full-pipeline
    ```
    """
    print("🚀 Running full mario.replays pipeline...")
    setup_env(c)
    setup_mario_dataset(c)
    create_replays(c)
    print("✅ Full pipeline complete!")