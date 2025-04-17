import os
import os.path as op
import pandas as pd
import json
import logging



def load_replay_data(replay_dir, type='metadata'):
    """
    Load the sidecar files for the clips in the specified directory.

    Args:
        clips_dir (str): The directory containing the clips and their sidecar files.
        type (str): The type of data to load. Default is 'metadata', can also be 'variables'.

    Returns:
        pandas.DataFrame: A dataframe where the rows are the individual clips and the columns are the sidecar data.
    """

    """Load sidecar files from a replay directory."""
    sidecars_data = []
    for root, folder, files in sorted(os.walk(replay_dir)):
        for file in files:
            if file.endswith(".json"):
                if type == 'metadata' and "infos" in root:
                    sidecars_files = op.join(root, file)
                    with open(sidecars_files) as f:
                        sidecars_data.append(json.load(f))
                elif type == 'variables' and "variables" in root:
                    sidecars_files = op.join(root, file)
                    with open(sidecars_files) as f:
                        sidecars_data.append(json.load(f))
    sidecars_df = pd.DataFrame(sidecars_data)
    return sidecars_df


def collect_bk2_files(DATA_PATH, subjects=None, sessions=None):
    """Collect all bk2 files and related information from the dataset."""
    bk2_files_info = []
    for root, _, files in sorted(os.walk(DATA_PATH)):
        # Skip undesired folders
        if "sourcedata" in root:
            continue

        for file in files:
            # Look for events.tsv that are not annotated
            if "events.tsv" in file and "annotated" not in file:
                run_events_file = op.join(root, file)
                logging.info(f"Processing events file: {file}")
                events_dataframe = pd.read_table(run_events_file)
                events_dataframe = events_dataframe[events_dataframe['trial_type'] == 'gym-retro_game']

                basename = op.basename(run_events_file)
                entities = basename.split('_')
                entities_dict = {}
                for ent in entities:
                    if '-' in ent:
                        key, value = ent.split('-', 1)
                        entities_dict[key] = value

                sub = entities_dict.get('sub')
                ses = entities_dict.get('ses')
                run = entities_dict.get('run')

                if not sub or not ses or not run:
                    logging.warning(f"Could not extract subject, session, or run from filename {basename}")
                    continue

                # Apply subject/session filters if specified
                if subjects and sub not in subjects:
                    continue
                if sessions and ses not in sessions:
                    continue

                # Gather the BK2 paths from the events file
                bk2_files = events_dataframe['stim_file'].values.tolist()
                for bk2_idx, bk2_file in enumerate(bk2_files):
                    if bk2_file != "Missing file" and not isinstance(bk2_file, float):
                        bk2_files_info.append({
                            'bk2_file': bk2_file,
                            'bk2_idx': bk2_idx,
                            'sub': sub,
                            'ses': ses,
                            'run': run
                        })

    return bk2_files_info