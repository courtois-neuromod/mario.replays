#!/usr/bin/env python
"""
This script is used to generate JSON sidecar files (and optionally additional files)
and playback videos for the mario dataset.
By default, only the JSON file is kept.
Use the flags below to have the script generate and save extra files:
  --save_video      : Save the playback video file (.mp4).
  --save_variables  : Save the variables file (.npz) that contains game variables.
  --save_states     : Save the full RAM state at each frame into a *_states.npy file.
  
Use the -v/--verbose flag to display verbose output.
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
import skvideo.io
from PIL import Image
from mario_replays.utils import replay_bk2, make_mp4, create_sidecar_dict



def format_repetition_variables(info_list, actions_list, buttons, bk2_file):
    """
    Build a repetition_variables dictionary from the collected per-frame info and actions,
    plus the list of button names.
    """
    repetition_variables = {}
    for key in info_list[0].keys():
        repetition_variables[key] = []
    for frame in info_list:
        for key in repetition_variables.keys():
            repetition_variables[key].append(frame[key])
    for idx, button in enumerate(buttons):
        repetition_variables[button] = []
        for action in actions_list:
            repetition_variables[button].append(action[idx])
    repetition_variables["filename"] = bk2_file
    try:
        repetition_variables["level"] = bk2_file.split("/")[-1].split("_")[-2].split("-")[1]
    except Exception:
        repetition_variables["level"] = None
    try:
        repetition_variables["subject"] = bk2_file.split("/")[-1].split("_")[0]
        repetition_variables["session"] = bk2_file.split("/")[-1].split("_")[1]
        repetition_variables["repetition"] = bk2_file.split("/")[-1].split("_")[-1].split(".")[0]
    except Exception:
        repetition_variables["subject"] = repetition_variables["session"] = repetition_variables["repetition"] = None
    repetition_variables["terminate"] = [True]  # Placeholder for termination info
    return repetition_variables

def get_passage_order(tasks):
    df = pd.DataFrame(tasks, columns=["bk2_file", "bk2_idx", "stimuli_path", "run", "save_video", "save_variables", "save_states"])
    df['subject'] = df['bk2_file'].str.extract(r'sub-(\d+)').astype(int)
    df['session'] = df['bk2_file'].str.extract(r'ses-(\d+)').astype(int)
    df['run'] = df['run'].str.extract(r'run-(\d+)').astype(int)
    df['bk2_idx'] = df['bk2_idx'].astype(int)
    df['order_value'] = df['session'] * 10000 + df['run'] * 100 + df['bk2_idx']
    df.sort_values(by=['subject', 'order_value'], inplace=True)
    df['passage_order'] = df.groupby('subject').cumcount()
    df.drop(columns=['subject', 'session', 'order_value'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return list(df.itertuples(index=False, name=None))


def process_bk2_file(task, DATA_PATH):
    """
    Process one .bk2 file.

    Parameters:
      task: a tuple (bk2_file, bk2_idx, stimuli_path, run, save_video, save_variables, save_states, total_idx)
    """
    bk2_file, bk2_idx, STIMULI_PATH, run, save_video, save_variables, save_states, total_idx = task
    bk2_path = op.abspath(op.join(DATA_PATH, bk2_file))
    if bk2_file == "Missing file" or isinstance(bk2_path, float):
        return
    if not op.exists(bk2_path):
        logging.error(f"File not found: {bk2_path}")
        return

    logging.debug(f"Adding stimuli path: {STIMULI_PATH}")
    retro.data.Integrations.add_custom_path(STIMULI_PATH)

    # Extract subject, session, and run from the bk2 filename.
    base = op.basename(bk2_file)
    parts = base.split("_")
    try:
        subject = parts[0]  # e.g., "sub-01"
        session = parts[1]  # e.g., "ses-01"
        run_part = parts[-1].replace(".bk2", "")
        if run_part.startswith("run-"):
            run_num = int(run_part.split("-")[-1])
            run_str = f"run-{run_num:02d}"
        else:
            run_str = f"run-{int(run):02d}"
    except Exception:
        subject = "sub-unknown"
        session = "ses-unknown"
        run_str = f"run-{int(run):02d}"
    # Create the output directory inside DATA_PATH/derivatives/bids/
    output_dir = op.join(DATA_PATH, "derivatives", "replays", subject, session)
    os.makedirs(output_dir, exist_ok=True)
    # Set the output file names using BIDS-like naming.
    json_sidecar_fname = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-info.json")
    # Check if already processed.
    if op.exists(json_sidecar_fname):
        logging.info(f"Already processed: {json_sidecar_fname}")
        return

    logging.info(f"Processing: {bk2_path}")
    

    # These file names will be re-used later.
    npz_file = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-variables.npz")
    video_file = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-video.mp4")
    states_file = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-states.npy")

    info_list = []
    actions_list = []
    frames_list = [] if save_video else None
    states_list = [] if save_states else None
    buttons = None

    for frame, keys, annotations, _, actions, state in replay_bk2(
        bk2_path, skip_first_step=(bk2_idx == 0)
    ):
        info_list.append(annotations["info"])
        actions_list.append(keys)
        if buttons is None:
            buttons = actions  # capture the button names
        if save_video:
            frames_list.append(frame)
        if save_states:
            states_list.append(state)

    if save_variables:
        np.savez(npz_file, info=info_list, actions=actions_list)
        logging.info(f"Variables saved to: {npz_file}")

    repetition_variables = format_repetition_variables(info_list, actions_list, buttons, bk2_file)
    # In case format_repetition_variables did not set these properly, use the ones we extracted.
    subject = repetition_variables.get("subject", subject)
    session = repetition_variables.get("session", session)
    if not subject.startswith("sub-"):
        subject = "sub-" + str(subject)
    if not session.startswith("ses-"):
        session = "ses-" + str(session)
    # Rebuild output directory (in case the subject/session info changed).
    output_dir = op.join(DATA_PATH, "derivatives", "bids", subject, session)
    os.makedirs(output_dir, exist_ok=True)
    json_sidecar_fname = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-info.json")
    npz_file = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-variables.npz")
    video_file = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-video.mp4")
    states_file = op.join(output_dir, f"{subject}_{session}_{run_str}_desc-states.npy")

    info_dict = create_sidecar_dict(repetition_variables)
    info_dict['bk2_idx'] = bk2_idx
    info_dict['run'] = run
    info_dict['total_idx'] = total_idx
    with open(json_sidecar_fname, "w") as f:
        json.dump(info_dict, f)
    logging.info(f"JSON saved for: {json_sidecar_fname}")

    if save_video:
        try:
            make_mp4(frames_list, video_file)
            logging.info(f"Video saved to: {video_file}")
        except Exception as e:
            logging.error(f"Could not write video file {video_file}: {e}")
    if save_states:
        np.save(states_file, np.array(states_list))
        logging.info(f"States saved to: {states_file}")


def main(args):
        # Set logging level based on --verbose flag.
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

        # Get datapath
    DATA_PATH = op.abspath(args.datapath)


    # Setup OUTPUT folder
    if args.output is None:
        OUTPUT_FOLDER = op.abspath(op.join(DATA_PATH, "derivatives"))
    else:
        OUTPUT_FOLDER = op.abspath(args.output)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Integrate game
    if args.stimuli is None:
        STIMULI_PATH = op.abspath(op.join(DATA_PATH, "stimuli"))
    else:
        STIMULI_PATH = op.abspath(args.stimuli)

    tasks = []
    for root, folders, files in sorted(os.walk(DATA_PATH)):
        if "sourcedata" in root:
            continue
        for file in files:
            if "events.tsv" in file and "annotated" not in file:
                run_events_file = op.join(root, file)
                run = file.split("_")[-2]
                logging.info(f"Processing events file: {run_events_file}")
                try:
                    events_dataframe = pd.read_table(run_events_file)
                except Exception as e:
                    logging.error(f"Cannot read {run_events_file}: {e}")
                    continue
                bk2_files = events_dataframe["stim_file"].values.tolist()
                for bk2_idx, bk2_file in enumerate(bk2_files):
                    if isinstance(bk2_file, str):
                        if '.bk2' in bk2_file:
                            tasks.append(
                                (
                                    bk2_file,
                                    bk2_idx,
                                    STIMULI_PATH,
                                    run,
                                    args.save_video,
                                    args.save_variables,
                                    args.save_states,
                                )
                            )
    tasks = get_passage_order(tasks)
    logging.info(f"Found {len(tasks)} bk2 files to process.")

    # Determine number of workers.
    n_jobs = os.cpu_count() if args.n_jobs == -1 else args.n_jobs

    if n_jobs != 1:
        with tqdm_joblib(tqdm(desc="Processing files", total=len(tasks))) as progress_bar:
            Parallel(n_jobs=n_jobs)(delayed(process_bk2_file)(task, DATA_PATH) for task in tasks)
    else:
        for task in tqdm(tasks, desc="Processing files"):
            process_bk2_file(task, DATA_PATH)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        default="data/mario",
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
        default=None,
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
        "--save_video",
        action="store_true",
        help="Save the playback video file (.mp4).",
    )
    parser.add_argument(
        "--save_variables",
        action="store_true",
        help="Save the variables file (.npz) that contains game variables.",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Save full RAM state at each frame into a *_states.npy file.",
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
