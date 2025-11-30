import retro
import skvideo
from skvideo import io
from PIL import Image
import os.path as op
import logging
import numpy as np
import logging
from mario_replays import replay_bk2
import os
import pandas as pd
from retro.enums import State

# Import general-purpose functions from cneuromod_vg_utils
from cneuromod_vg_utils.replay import get_variables_from_replay as _get_variables_from_replay_general
from cneuromod_vg_utils.video import make_gif, make_mp4, make_webp

# ===============================
# 🔹 GAME VARIABLES MANIPULATION
# ===============================


def get_variables_from_replay(
    bk2_fpath,
    skip_first_step=True,
    state=State.DEFAULT,
    game=None,
    scenario=None,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
    """
    Replay the bk2 file and return game variables and frames.

    This function now uses the general-purpose implementation from cneuromod_vg_utils,
    but adapts the output to match the original mario.replays signature (without audio).
    """
    repetition_variables, replay_info, replay_frames, replay_states, audio_track, audio_rate = _get_variables_from_replay_general(
        bk2_fpath,
        skip_first_step=skip_first_step,
        state=state,
        game=game,
        scenario=scenario,
        inttype=inttype,
    )

    # Return without audio data to maintain backwards compatibility
    return repetition_variables, replay_info, replay_frames, replay_states


def reformat_info(info, keys, bk2_fpath, actions):
    """Create a structured dictionary from replay info."""
    filename = op.basename(bk2_fpath)
    entities = filename.split("_")
    entities_dict = {}
    for ent in entities:
        if "-" in ent:
            key, value = ent.split("-", 1)
            entities_dict[key] = value

    repetition_variables = {
        "filename": bk2_fpath,
        "level": entities_dict.get("level"),
        "subject": entities_dict.get("sub"),
        "session": entities_dict.get("ses"),
        "actions": actions,
    }

    for key in info[0].keys():
        repetition_variables[key] = []
    for button in actions:
        repetition_variables[button] = []

    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(actions):
            repetition_variables[button].append(keys[frame_idx][button_idx])
    return repetition_variables


def create_sidecar_dict(repetition_variables):
    sidecar_dict = {}
    sidecar_dict["Subject"] = repetition_variables["subject"]
    sidecar_dict["World"] = repetition_variables["level"][1]
    sidecar_dict["Level"] = repetition_variables["level"][-1]
    sidecar_dict["Duration"] = len(repetition_variables["score"]) / 60
    # sidecar_dict["Terminated"] = repetition_variables["terminate"][-1] == True
    if repetition_variables["player_y_screen"][-1] > 1:
        cleared = False
    elif repetition_variables["lives"][-1] == -1:
        cleared = False
    elif repetition_variables["player_state"][-1] == 6:
        cleared = False
    elif repetition_variables["player_state"][-1] == 11:
        cleared = False
    else:
        cleared = True

    sidecar_dict["Cleared"] = cleared
    sidecar_dict["ScoreGained"] = (
        repetition_variables["score"][-1] - repetition_variables["score"][0]
    )
    sidecar_dict["X_Traveled"] = (
        repetition_variables["xscrollLo"][-1]
        + (256 * repetition_variables["xscrollHi"][-1])
    ) - (
        repetition_variables["xscrollLo"][0]
        + (256 * repetition_variables["xscrollHi"][0])
    )
    sidecar_dict["Average_speed"] = (
        sidecar_dict["X_Traveled"] / sidecar_dict["Duration"]
    )
    sidecar_dict["Lives_lost"] = (
        repetition_variables["lives"][0] - repetition_variables["lives"][-1]
    )
    sidecar_dict["Hits_taken"] = count_hits_taken(repetition_variables)
    sidecar_dict["Enemies_killed"] = count_kills(repetition_variables)
    sidecar_dict["Powerups_collected"] = count_powerups_collected(repetition_variables)
    sidecar_dict["Bricks_destroyed"] = count_bricks_destroyed(repetition_variables)
    sidecar_dict["CoinsGained"] = (
        repetition_variables["coins"][-1] - repetition_variables["coins"][0]
    )
    return sidecar_dict


# ---------------------------
# GET GAME STATS
# ---------------------------


def count_kills(repetition_variables):
    kill_count = 0
    for i in range(6):
        for idx, val in enumerate(repetition_variables[f"enemy_kill3{i}"][:-1]):
            if val in [4, 34, 132]:
                if repetition_variables[f"enemy_kill3{i}"][idx + 1] != val:
                    if i == 5:
                        if repetition_variables["powerup_yes_no"] == 0:
                            kill_count += 1
                    else:
                        kill_count += 1
    return kill_count


def count_bricks_destroyed(repetition_variables):
    score_increments = list(np.diff(repetition_variables["score"]))
    bricks_destroyed = 0
    for idx, inc in enumerate(score_increments):
        if inc == 5:
            if repetition_variables["jump_airborne"][idx] == 1:
                bricks_destroyed += 1
    return bricks_destroyed


def count_hits_taken(repetition_variables):
    diff_state = list(np.diff(repetition_variables["powerstate"]))
    hits_count = 0
    for idx, val in enumerate(diff_state):
        if val < -10000:
            hits_count += 1
    diff_lives = list(np.diff(repetition_variables["lives"]))
    for idx, val in enumerate(diff_lives):
        if val < 0:
            hits_count += 1
    return hits_count


def count_powerups_collected(repetition_variables):
    powerup_count = 0
    for idx, val in enumerate(repetition_variables["player_state"][:-1]):
        if val in [9, 12, 13]:
            if repetition_variables["player_state"][idx + 1] != val:
                powerup_count += 1
    return powerup_count


# ===============================
# 🔹 FILES CREATION
# ===============================
# Video creation functions (make_gif, make_mp4, make_webp) are now imported
# from cneuromod_vg_utils.video instead of being defined here.
