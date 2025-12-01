"""
This module now uses the general-purpose replay_bk2 function from cneuromod_vg_utils.
The function signature has been updated to match the new version.

For backwards compatibility, we provide a wrapper that adapts the new signature
to the old one used in mario-specific code.
"""

from cneuromod_vg_utils.replay import replay_bk2 as _replay_bk2_new
from retro.enums import State
import retro


def replay_bk2(
    bk2_path, skip_first_step=False, state=State.DEFAULT, game=None, scenario=None, inttype=retro.data.Integrations.CUSTOM_ONLY
):
    """
    Create an iterator that replays a bk2 file, yielding frame, keys, annotations, truncate, actions, and state.

    This is a wrapper around cneuromod_vg_utils.replay.replay_bk2 that adapts the output
    to match the original mario.replays signature (without audio).

    Args:
        bk2_path (str): Path to the bk2 file to be replayed.
        skip_first_step (bool, optional): Whether to skip the first step of the movie. Defaults to False. For CNeuroMod data, apply to first bk2 of each run.
        game (str, optional): The name of the game. If None, it will be inferred from the bk2 file. Defaults to None.
        scenario (str, optional): The scenario to be used in the emulator. Defaults to None.
        inttype (retro.data.Integrations, optional): The integration type for the emulator. Defaults to retro.data.Integrations.CUSTOM_ONLY.

    Yields:
        tuple: A tuple containing:
            - frame (numpy.ndarray): The current frame of the game.
            - keys (list): The list of keys pressed by the players.
            - annotations (dict): A dictionary containing reward, done, and info.
            - truncate (bool): Whether the episode was truncated.
            - actions (list): The list of possible actions in the game.
            - state (bytes): The current state of the emulator.
    """
    for frame, keys, annotations, audio_chunk, audio_rate, truncate, actions, state in _replay_bk2_new(
        bk2_path,
        skip_first_step=skip_first_step,
        state=state,
        game=game,
        scenario=scenario,
        inttype=inttype,
    ):
        # Yield without audio data to match original signature
        yield frame, keys, annotations, truncate, actions, state