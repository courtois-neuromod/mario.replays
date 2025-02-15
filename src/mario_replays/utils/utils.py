import retro
import skvideo
from PIL import Image

def replay_bk2(
    bk2_path, skip_first_step=True, game=None, scenario=None, stimuli_path=None, 
):
    """Create an iterator that replays a bk2 file, yielding frame, keys, annotations, sound, actions, and state."""
    import logging
    movie = retro.Movie(bk2_path)
    if game is None:
        game = movie.get_game()
    logging.debug(f"Creating emulator for game: {game}")

    if stimuli_path is not None:
        retro.data.Integrations.add_custom_path(stimuli_path)
        inttype=retro.data.Integrations.CUSTOM_ONLY
    else:
        inttype=retro.data.Integrations.ALL

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


