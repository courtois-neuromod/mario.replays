import retro
import skvideo
from PIL import Image
import os.path as op
import logging
import numpy as np
import logging
from mario_replays import replay_bk2


# ===============================
# 🔹 GAME VARIABLES MANIPULATION
# ===============================

def get_variables_from_replay(
    bk2_fpath, skip_first_step=True, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """Replay the bk2 file and return game variables and frames."""
    replay = replay_bk2(
        bk2_fpath, skip_first_step=skip_first_step, game=game, scenario=scenario, inttype=inttype
    )
    all_frames = []
    all_keys = []
    all_info = []
    annotations = {}

    for frame, keys, annotations, _, actions, _ in replay:
        all_keys.append(keys)
        all_info.append(annotations["info"])
        all_frames.append(frame)


    repetition_variables = reformat_info(all_info, all_keys, bk2_fpath, actions)

    if not annotations.get('done', False):
        logging.warning(f"Done condition not satisfied for {bk2_fpath}. Consider changing skip_first_step.")

    return repetition_variables, all_frames


def reformat_info(info, keys, bk2_fpath, actions):
    """Create a structured dictionary from replay info."""
    filename = op.basename(bk2_fpath)
    entities = filename.split('_')
    entities_dict = {}
    for ent in entities:
        if '-' in ent:
            key, value = ent.split('-', 1)
            entities_dict[key] = value

    repetition_variables = {
        "filename": bk2_fpath,
        "level": entities_dict.get('level'),
        "subject": entities_dict.get('sub'),
        "session": entities_dict.get('ses'),
        "repetition": entities_dict.get('run'),
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
    sidecar_dict["world"] = repetition_variables["level"][1]
    sidecar_dict["level"] = repetition_variables["level"][-1]
    sidecar_dict["duration"] = len(repetition_variables["score"]) / 60
    sidecar_dict["terminated"] = repetition_variables["terminate"][-1] == True
    sidecar_dict["cleared"] = all([repetition_variables["terminate"][-1] == True, repetition_variables["lives"][-1] >= 0])
    sidecar_dict["final_score"] = repetition_variables["score"][-1]
    sidecar_dict["final_position"] = repetition_variables["xscrollLo"][-1] + (256 * repetition_variables["xscrollHi"][-1])
    sidecar_dict["lives_lost"] = 2 - repetition_variables["lives"][-1]
    sidecar_dict["hits_taken"] = count_hits_taken(repetition_variables)
    sidecar_dict["enemies_killed"] = count_kills(repetition_variables)
    sidecar_dict["powerups_collected"] = count_powerups_collected(repetition_variables)
    sidecar_dict["bricks_destroyed"] = count_bricks_destroyed(repetition_variables)
    sidecar_dict["coins"] = repetition_variables["coins"][-1]
    return sidecar_dict

# ---------------------------
# UTILITY FUNCTIONS
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

def make_gif(selected_frames, movie_fname):
    """Create a GIF file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname, save_all=True, append_images=frame_list[1:], optimize=False, duration=16, loop=0
    )


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


def make_webp(selected_frames, movie_fname):
    """Create a WebP file from a list of frames."""
    frame_list = [Image.fromarray(np.uint8(img), "RGB") for img in selected_frames]

    if not frame_list:
        logging.warning(f"No frames to save in {movie_fname}")
        return

    frame_list[0].save(
        movie_fname, 'WEBP', quality=50, lossless=False, save_all=True, append_images=frame_list[1:], duration=16, loop=0
    )

