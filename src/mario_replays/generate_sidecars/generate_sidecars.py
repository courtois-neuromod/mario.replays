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

# ---------------------------
# PARSER ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--datapath",
    default=".",
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
# Default n_jobs is -1 (use all available cores)
parser.add_argument(
    "-j",
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

# Set logging level based on --verbose flag.
if args.verbose:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
else:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def count_kills(repvars):
    kill_count = 0
    for i in range(6):
        for idx, val in enumerate(repvars[f"enemy_kill3{i}"][:-1]):
            if val in [4, 34, 132]:
                if repvars[f"enemy_kill3{i}"][idx + 1] != val:
                    if i == 5:
                        if repvars["powerup_yes_no"] == 0:
                            kill_count += 1
                    else:
                        kill_count += 1
    return kill_count

def count_bricks_destroyed(repvars):
    score_increments = list(np.diff(repvars["score"]))
    bricks_destroyed = 0
    for idx, inc in enumerate(score_increments):
        if inc == 5:
            if repvars["jump_airborne"][idx] == 1:
                bricks_destroyed += 1
    return bricks_destroyed

def count_hits_taken(repvars):
    diff_state = list(np.diff(repvars["powerstate"]))
    hits_count = 0
    for idx, val in enumerate(diff_state):
        if val < -10000:
            hits_count += 1
    diff_lives = list(np.diff(repvars["lives"]))
    for idx, val in enumerate(diff_lives):
        if val < 0:
            hits_count += 1
    return hits_count

def count_powerups_collected(repvars):
    powerup_count = 0
    for idx, val in enumerate(repvars["player_state"][:-1]):
        if val in [9, 12, 13]:
            if repvars["player_state"][idx + 1] != val:
                powerup_count += 1
    return powerup_count

def create_info_dict(repvars):
    info_dict = {}
    info_dict["world"] = repvars["level"][1]
    info_dict["level"] = repvars["level"][-1]
    info_dict["duration"] = len(repvars["score"]) / 60
    info_dict["terminated"] = repvars["terminate"][-1] == True
    info_dict["cleared"] = all([repvars["terminate"][-1] == True, repvars["lives"][-1] >= 0])
    info_dict["final_score"] = repvars["score"][-1]
    info_dict["final_position"] = repvars["xscrollLo"][-1] + (256 * repvars["xscrollHi"][-1])
    info_dict["lives_lost"] = 2 - repvars["lives"][-1]
    info_dict["hits_taken"] = count_hits_taken(repvars)
    info_dict["enemies_killed"] = count_kills(repvars)
    info_dict["powerups_collected"] = count_powerups_collected(repvars)
    info_dict["bricks_destroyed"] = count_bricks_destroyed(repvars)
    info_dict["coins"] = repvars["coins"][-1]
    return info_dict

def format_repvars(info_list, actions_list, buttons, bk2_file):
    """
    Build a repvars dictionary from the collected per-frame info and actions,
    plus the list of button names.
    """
    repvars = {}
    for key in info_list[0].keys():
        repvars[key] = []
    for frame in info_list:
        for key in repvars.keys():
            repvars[key].append(frame[key])
    for idx, button in enumerate(buttons):
        repvars[button] = []
        for action in actions_list:
            repvars[button].append(action[idx])
    repvars["filename"] = bk2_file
    try:
        repvars["level"] = bk2_file.split("/")[-1].split("_")[-2].split("-")[1]
    except Exception:
        repvars["level"] = None
    try:
        repvars["subject"] = bk2_file.split("/")[-1].split("_")[0]
        repvars["session"] = bk2_file.split("/")[-1].split("_")[1]
        repvars["repetition"] = bk2_file.split("/")[-1].split("_")[-1].split(".")[0]
    except Exception:
        repvars["subject"] = repvars["session"] = repvars["repetition"] = None
    repvars["terminate"] = [True]  # Placeholder for termination info
    return repvars


def replay_bk2(
    bk2_path, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Create an iterator that replays a bk2 file, yielding frame, keys, annotations, sound, actions, and state."""
    import logging
    movie = retro.Movie(bk2_path)
    if game is None:
        game = movie.get_game()
    logging.debug(f"Creating emulator for game: {game}")
    emulator = retro.make(game, scenario=scenario, inttype=inttype, render_mode=False)
    emulator.initial_state = movie.get_state()
    actions = emulator.buttons
    emulator.reset()
    if skip_first_step:
        movie.step()
    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(emulator.num_buttons):
                keys.append(movie.get_key(i, p))
        frame, rew, terminate, truncate, info = emulator.step(keys)
        annotations = {"reward": rew, "done": terminate, "info": info}
        state = emulator.em.get_state()
        yield frame, keys, annotations, None, actions, state
    emulator.close()
    movie.close()

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


def make_mp4(selected_frames, movie_fname):
    """Create an MP4 file from a list of frames."""
    writer = skvideo.io.FFmpegWriter(
        movie_fname, inputdict={"-r": "60"}, outputdict={"-r": "60"}
    )
    for frame in selected_frames:
        im = Image.new("RGB", (frame.shape[1], frame.shape[0]), color="white")
        im.paste(Image.fromarray(frame), (0, 0))
        writer.writeFrame(np.array(im))
    writer.close()

# ---------------------------
# PER-FILE PROCESSING FUNCTION
# ---------------------------
def process_bk2_file(task):
    """
    Process one .bk2 file.

    Parameters:
      task: a tuple (bk2_file, bk2_idx, stimuli_path, save_video, save_variables, save_states)
    """
    bk2_file, bk2_idx, stimuli_path, run, save_video, save_variables, save_states, total_idx = task

    if bk2_file == "Missing file" or isinstance(bk2_file, float):
        return
    if not op.exists(bk2_file):
        logging.error(f"File not found: {bk2_file}")
        return
    if op.exists(bk2_file.replace(".bk2", ".json")):
        logging.info(f"Already processed: {bk2_file}")
        return

    try:
        logging.info(f"Processing: {bk2_file}")
        retro.data.Integrations.add_custom_path(stimuli_path)

        npz_file = bk2_file.replace(".bk2", ".npz")
        video_file = bk2_file.replace(".bk2", ".mp4")

        info_list = []
        actions_list = []
        frames_list = [] if save_video else None
        states_list = [] if save_states else None
        buttons = None

        for frame, keys, annotations, _, actions, state in replay_bk2(
            bk2_file, skip_first_step=(bk2_idx == 0)
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

        repvars = format_repvars(info_list, actions_list, buttons, bk2_file)
        info_dict = create_info_dict(repvars)
        info_dict['bk2_idx'] = bk2_idx
        info_dict['run'] = run
        info_dict['total_idx'] = total_idx
        json_sidecar_fname = bk2_file.replace(".bk2", ".json")
        with open(json_sidecar_fname, "w") as f:
            json.dump(info_dict, f)
        logging.info(f"JSON saved for: {bk2_file}")

        if save_video:
            try:
                make_mp4(frames_list, video_file)
                logging.info(f"Video saved to: {video_file}")
            except Exception as e:
                logging.error(f"Could not write video file {video_file}: {e}")
        if save_states:
            states_file = bk2_file.replace(".bk2", "_states.npy")
            np.save(states_file, np.array(states_list))
            logging.info(f"States saved to: {states_file}")

    except Exception as e:
        logging.error(f"Exception processing {bk2_file}: {e}")

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main(args):
    DATA_PATH = args.datapath
    if DATA_PATH == ".":
        logging.info("No data path specified. Searching files in this folder.")
    logging.info(f"Generating annotations for the mario dataset in: {DATA_PATH}")

    if args.stimuli is None:
        stimuli_path = op.join(os.getcwd(), "stimuli")
        logging.info(f"Using stimuli path: {stimuli_path}")
    else:
        stimuli_path = op.join(args.stimuli)
        logging.info(f"Using stimuli path: {stimuli_path}")

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
                                    stimuli_path,
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
            Parallel(n_jobs=n_jobs)(delayed(process_bk2_file)(task) for task in tasks)
    else:
        for task in tqdm(tasks, desc="Processing files"):
            process_bk2_file(task)

if __name__ == "__main__":
    main(args)
