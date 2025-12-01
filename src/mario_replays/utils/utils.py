"""Utility functions for Mario replay processing."""

import retro
import os.path as op
import numpy as np
from retro.enums import State

from cneuromod_vg_utils.replay import get_variables_from_replay as _get_variables_from_replay_general
from cneuromod_vg_utils.video import make_gif, make_mp4, make_webp


def get_variables_from_replay(
    bk2_fpath,
    skip_first_step=True,
    state=State.DEFAULT,
    game=None,
    scenario=None,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
    """
    Replay a bk2 file and extract game variables and frames.

    Adapts general-purpose implementation from cneuromod_vg_utils
    to maintain backwards compatibility without audio data.

    Args:
        bk2_fpath: Path to .bk2 replay file
        skip_first_step: Skip first frame (for CNeuroMod data)
        state: Emulator state
        game: Game name (inferred if None)
        scenario: Emulator scenario
        inttype: Integration type

    Returns:
        Tuple of (repetition_variables, replay_info, replay_frames, replay_states)
    """
    result = _get_variables_from_replay_general(
        bk2_fpath, skip_first_step=skip_first_step, state=state,
        game=game, scenario=scenario, inttype=inttype
    )
    repetition_variables, replay_info, replay_frames, replay_states, _, _ = result
    return repetition_variables, replay_info, replay_frames, replay_states


def _extract_entities_from_filename(filename):
    """Extract BIDS-like entities from filename."""
    entities = filename.split("_")
    entities_dict = {}
    for ent in entities:
        if "-" in ent:
            key, value = ent.split("-", 1)
            entities_dict[key] = value
    return entities_dict


def _build_replay_metadata(bk2_fpath, actions):
    """Build base metadata dictionary from replay file path."""
    filename = op.basename(bk2_fpath)
    entities_dict = _extract_entities_from_filename(filename)

    return {
        "filename": bk2_fpath,
        "level": entities_dict.get("level"),
        "subject": entities_dict.get("sub"),
        "session": entities_dict.get("ses"),
        "actions": actions,
    }


def _add_frame_data_to_dict(repetition_variables, info, keys, actions):
    """Add per-frame game info and button presses to variables dict."""
    for key in info[0].keys():
        repetition_variables[key] = []
    for button in actions:
        repetition_variables[button] = []

    for frame_idx, frame_info in enumerate(info):
        for key in frame_info.keys():
            repetition_variables[key].append(frame_info[key])
        for button_idx, button in enumerate(actions):
            repetition_variables[button].append(keys[frame_idx][button_idx])


def reformat_info(info, keys, bk2_fpath, actions):
    """Create structured dictionary from replay info."""
    repetition_variables = _build_replay_metadata(bk2_fpath, actions)
    _add_frame_data_to_dict(repetition_variables, info, keys, actions)
    return repetition_variables


def _calculate_world_and_level(level_str):
    """Extract world and level numbers from level string."""
    return level_str[1], level_str[-1]


def _calculate_distance_traveled(repetition_variables):
    """Calculate total X distance traveled."""
    start_x = (repetition_variables["xscrollLo"][0] +
               (256 * repetition_variables["xscrollHi"][0]))
    end_x = (repetition_variables["xscrollLo"][-1] +
             (256 * repetition_variables["xscrollHi"][-1]))
    return end_x - start_x


def _check_level_cleared(repetition_variables):
    """Determine if level was successfully cleared."""
    if repetition_variables["player_y_screen"][-1] > 1:
        return False
    if repetition_variables["lives"][-1] == -1:
        return False
    if repetition_variables["player_state"][-1] in [6, 11]:
        return False
    return True


def create_sidecar_dict(repetition_variables):
    """
    Create JSON sidecar metadata from replay variables.

    Extracts high-level statistics from frame-by-frame game data.

    Args:
        repetition_variables: Dictionary with per-frame game variables

    Returns:
        Dictionary with summary statistics for the replay
    """
    world, level = _calculate_world_and_level(repetition_variables["level"])
    duration = len(repetition_variables["score"]) / 60
    distance = _calculate_distance_traveled(repetition_variables)

    return {
        "Subject": repetition_variables["subject"],
        "World": world,
        "Level": level,
        "Duration": duration,
        "Cleared": _check_level_cleared(repetition_variables),
        "ScoreGained": repetition_variables["score"][-1] - repetition_variables["score"][0],
        "X_Traveled": distance,
        "Average_speed": distance / duration,
        "Lives_lost": repetition_variables["lives"][0] - repetition_variables["lives"][-1],
        "Hits_taken": count_hits_taken(repetition_variables),
        "Enemies_killed": count_kills(repetition_variables),
        "Powerups_collected": count_powerups_collected(repetition_variables),
        "Bricks_destroyed": count_bricks_destroyed(repetition_variables),
        "CoinsGained": repetition_variables["coins"][-1] - repetition_variables["coins"][0],
    }


def _count_enemy_kills_for_slot(repetition_variables, slot_idx):
    """Count kills for a specific enemy slot."""
    kill_count = 0
    enemy_key = f"enemy_kill3{slot_idx}"

    for idx, val in enumerate(repetition_variables[enemy_key][:-1]):
        if val in [4, 34, 132]:
            if repetition_variables[enemy_key][idx + 1] != val:
                if slot_idx == 5 and repetition_variables["powerup_yes_no"] == 0:
                    kill_count += 1
                elif slot_idx != 5:
                    kill_count += 1
    return kill_count


def count_kills(repetition_variables):
    """Count total enemies killed during replay."""
    return sum(_count_enemy_kills_for_slot(repetition_variables, i) for i in range(6))


def count_bricks_destroyed(repetition_variables):
    """Count bricks destroyed by jumping."""
    score_increments = list(np.diff(repetition_variables["score"]))
    bricks_destroyed = 0

    for idx, inc in enumerate(score_increments):
        if inc == 5 and repetition_variables["jump_airborne"][idx] == 1:
            bricks_destroyed += 1
    return bricks_destroyed


def _count_powerstate_hits(repetition_variables):
    """Count hits from powerstate changes."""
    diff_state = list(np.diff(repetition_variables["powerstate"]))
    return sum(1 for val in diff_state if val < -10000)


def _count_life_losses(repetition_variables):
    """Count hits from life losses."""
    diff_lives = list(np.diff(repetition_variables["lives"]))
    return sum(1 for val in diff_lives if val < 0)


def count_hits_taken(repetition_variables):
    """Count total hits taken (damage + deaths)."""
    return _count_powerstate_hits(repetition_variables) + _count_life_losses(repetition_variables)


def count_powerups_collected(repetition_variables):
    """Count powerups collected during replay."""
    powerup_count = 0

    for idx, val in enumerate(repetition_variables["player_state"][:-1]):
        if val in [9, 12, 13]:
            if repetition_variables["player_state"][idx + 1] != val:
                powerup_count += 1
    return powerup_count
