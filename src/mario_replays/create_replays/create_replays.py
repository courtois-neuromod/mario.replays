#!/usr/bin/env python
"""
Generate JSON sidecar files and optional outputs for Mario dataset replays.

By default, only JSON metadata files are created.
Optional flags enable saving videos, variables, and RAM dumps.

Usage:
    python create_replays.py --datapath sourcedata/mario --save_videos --verbose
"""

import argparse
import os
import os.path as op
import retro
import pandas as pd
import json
import numpy as np
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import logging
from mario_replays.utils import make_mp4, create_sidecar_dict, get_variables_from_replay


def _extract_subject_from_bk2(bk2_file):
    """Extract subject ID from bk2 filename."""
    return bk2_file.split("/")[-1].split("_")[0]


def _extract_session_from_bk2(bk2_file):
    """Extract session ID from bk2 filename."""
    return bk2_file.split("/")[-1].split("_")[1]


def _extract_level_from_bk2(bk2_file):
    """Extract level ID from bk2 filename."""
    return bk2_file.split("/")[-1].split("_")[3].split("-")[1]


def get_passage_order(bk2_df):
    """
    Sort replays and assign global and level-specific indices.

    Args:
        bk2_df: DataFrame with replay data including 'bk2_file' column

    Returns:
        DataFrame with added subject, session, level, global_idx, and level_idx columns
    """
    bk2_df["subject"] = [_extract_subject_from_bk2(x) for x in bk2_df["bk2_file"].values]
    bk2_df["session"] = [_extract_session_from_bk2(x) for x in bk2_df["bk2_file"].values]
    bk2_df["level"] = [_extract_level_from_bk2(x) for x in bk2_df["bk2_file"].values]

    bk2_df = bk2_df.sort_values(["subject", "session", "run", "idx_in_run"]).assign(
        global_idx=lambda x: x.groupby("subject").cumcount()
    )
    bk2_df = bk2_df.sort_values(
        ["subject", "level", "session", "run", "idx_in_run"]
    ).assign(level_idx=lambda x: x.groupby(["subject", "level"]).cumcount())
    return bk2_df.sort_values(["subject", "global_idx"])


def _setup_game_config(args):
    """Configure game name and output folder based on simple flag."""
    if args.simple:
        args.game_name = "SuperMarioBrosSimple-Nes"
        args.output_name = "replays_simple"
    else:
        args.game_name = "SuperMarioBros-Nes"
        args.output_name = "replays"


def _setup_stimuli_path(args, data_path):
    """Set up and register stimuli path with retro."""
    if args.stimuli is None:
        stimuli_path = op.abspath(op.join(data_path, "mario.stimuli"))
    else:
        stimuli_path = op.abspath(args.stimuli)
    logging.debug(f"Adding stimuli path: {stimuli_path}")
    retro.data.Integrations.add_custom_path(stimuli_path)


def _validate_bk2_file(bk2_file, bk2_path):
    """Check if bk2 file is valid and exists."""
    if bk2_file == "Missing file" or isinstance(bk2_path, float):
        return False
    if not op.exists(bk2_path):
        logging.error(f"File not found: {bk2_path}")
        return False
    return True


def _build_output_paths(output_folder, bk2_file, subject, session):
    """Build all output file paths for replay processing."""
    entities = bk2_file.split("/")[-1].split(".")[0]
    beh_folder = op.join(output_folder, subject, session, "beh")

    return {
        "mp4": op.join(beh_folder, "videos", f"{entities}.mp4"),
        "ramdump": op.join(beh_folder, "ramdumps", f"{entities}.npz"),
        "json": op.join(beh_folder, "infos", f"{entities}.json"),
        "variables": op.join(beh_folder, "variables", f"{entities}.json"),
        "entities": entities
    }


def _save_optional_outputs(args, paths, replay_frames, replay_states, repetition_variables):
    """Save video, ramdump, and variables files if requested."""
    if args.save_videos:
        os.makedirs(os.path.dirname(paths["mp4"]), exist_ok=True)
        make_mp4(replay_frames, paths["mp4"])
        logging.info(f"Video saved to: {paths['mp4']}")

    if args.save_ramdumps:
        os.makedirs(os.path.dirname(paths["ramdump"]), exist_ok=True)
        np.savez(paths["ramdump"], np.array(replay_states))
        logging.info(f"States saved to: {paths['ramdump']}")

    if args.save_variables:
        os.makedirs(os.path.dirname(paths["variables"]), exist_ok=True)
        with open(paths["variables"], "w") as f:
            json.dump(repetition_variables, f)


def _create_and_save_sidecar(repetition_variables, task_metadata, paths):
    """Create and save JSON sidecar with replay metadata."""
    info_dict = create_sidecar_dict(repetition_variables)
    info_dict.update({
        "IndexInRun": task_metadata["idx_in_run"],
        "Run": task_metadata["run"],
        "IndexGlobal": task_metadata["global_idx"],
        "IndexLevel": task_metadata["level_idx"],
        "Phase": task_metadata["phase"],
        "LevelFullName": task_metadata["level"],
        "Bk2File": paths["entities"]
    })

    os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
    with open(paths["json"], "w") as f:
        json.dump(info_dict, f)
    logging.info(f"JSON saved for: {paths['json']}")


def process_bk2_file(task, args):
    """
    Process a single .bk2 replay file.

    Extracts game data and creates JSON metadata sidecar.
    Optionally saves video, variables, and RAM dumps.

    Args:
        task: Tuple of (bk2_file, run, idx_in_run, phase, subject,
              session, level, global_idx, level_idx)
        args: Command-line arguments with processing options
    """
    _setup_game_config(args)
    data_path = op.abspath(args.datapath)
    output_folder = op.abspath(op.join(args.output, args.output_name))
    os.makedirs(output_folder, exist_ok=True)
    _setup_stimuli_path(args, data_path)

    bk2_file, run, idx_in_run, phase, subject, session, level, global_idx, level_idx = task
    bk2_path = op.abspath(op.join(data_path, bk2_file))

    if not _validate_bk2_file(bk2_file, bk2_path):
        return

    json_check_path = op.join(output_folder, bk2_file.replace(".bk2", ".json"))
    if op.exists(json_check_path):
        logging.info(f"Already processed: {json_check_path}")
        return

    logging.info(f"Processing: {bk2_path}")
    paths = _build_output_paths(output_folder, bk2_file, subject, session)

    repetition_variables, _, replay_frames, replay_states = get_variables_from_replay(
        op.join(data_path, bk2_file),
        skip_first_step=(idx_in_run == 0),
        game=args.game_name,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
    )

    _save_optional_outputs(args, paths, replay_frames, replay_states, repetition_variables)

    task_metadata = {
        "idx_in_run": idx_in_run, "run": run, "global_idx": global_idx,
        "level_idx": level_idx, "phase": phase, "level": level
    }
    _create_and_save_sidecar(repetition_variables, task_metadata, paths)


def _configure_logging(verbose):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")


def _determine_phase(events_dataframe):
    """Determine if replay is discovery or practice phase."""
    unique_levels = len(np.unique(events_dataframe["level"].dropna()))
    return "discovery" if unique_levels == 1 else "practice"


def _extract_run_from_filename(filename):
    """Extract run ID from events file name."""
    return filename.split("_")[-2]


def _collect_bk2_info_from_events(run_events_file):
    """Collect bk2 file info from a single events file."""
    run = _extract_run_from_filename(op.basename(run_events_file))
    logging.info(f"Processing events file: {run_events_file}")

    try:
        events_df = pd.read_table(run_events_file)
    except Exception as e:
        logging.error(f"Cannot read {run_events_file}: {e}")
        return []

    phase = _determine_phase(events_df)
    bk2_files = events_df["stim_file"].values.tolist()

    bk2_list = []
    for idx_in_run, bk2_file in enumerate(bk2_files):
        if isinstance(bk2_file, str) and ".bk2" in bk2_file:
            bk2_list.append({
                "bk2_file": bk2_file,
                "run": run,
                "idx_in_run": idx_in_run,
                "phase": phase,
            })
    return bk2_list


def _collect_all_bk2_files(data_path):
    """Walk dataset and collect all bk2 file information."""
    bk2_list = []
    for root, _, files in sorted(os.walk(data_path)):
        for file in files:
            if "events.tsv" in file and "annotated" not in file:
                run_events_file = op.join(root, file)
                bk2_list.extend(_collect_bk2_info_from_events(run_events_file))
    return bk2_list


def _run_parallel_processing(tasks, args):
    """Process tasks in parallel using joblib."""
    with tqdm_joblib(tqdm(desc="Processing files", total=len(tasks))):
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_bk2_file)(task, args) for task in tasks
        )


def _run_sequential_processing(tasks, args):
    """Process tasks sequentially with progress bar."""
    for task in tqdm(tasks, desc="Processing files"):
        process_bk2_file(task, args)


def main(args):
    """
    Main entry point for replay processing.

    Scans dataset for events files, collects bk2 file info,
    and processes each replay in parallel or sequentially.

    Args:
        args: Parsed command-line arguments
    """
    _configure_logging(args.verbose)
    data_path = op.abspath(args.datapath)

    bk2_list = _collect_all_bk2_files(data_path)
    bk2_df = pd.DataFrame(bk2_list)
    bk2_df = get_passage_order(bk2_df)

    tasks = [tuple(row) for row in bk2_df.values]
    logging.info(f"Found {len(tasks)} bk2 files to process.")

    n_jobs = os.cpu_count() if args.n_jobs == -1 else args.n_jobs
    if n_jobs != 1:
        _run_parallel_processing(tasks, args)
    else:
        _run_sequential_processing(tasks, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        default="sourcedata/mario",
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the mario dataset.",
    )
    parser.add_argument(
        "-s",
        "--stimuli",
        default=None,
        type=str,
        help="Data path to look for the stimuli files (rom, state files, data.json etc...).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='outputdata/',
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        default=-1,
        type=int,
        help="Number of parallel jobs to run. Use -1 to use all available cores.",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save the playback video file (.mp4).",
    )
    parser.add_argument(
        "--save_variables",
        action="store_true",
        help="Save the variables file (.json) that contains game variables.",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Save full RAM state at each frame into a *_states.npy file.",
    )
    parser.add_argument(
        "--save_ramdumps",
        action="store_true",
        help="Save RAM dumps at each frame into a *_ramdumps.npy file.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="If set, use the simplified game version (SuperMarioBrosSimple-Nes) "
        "and output into 'mario_scenes_simple' subfolder instead of 'mario_scenes'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verbose output.",
    )

    args = parser.parse_args()

    # Main loop
    main(args)
