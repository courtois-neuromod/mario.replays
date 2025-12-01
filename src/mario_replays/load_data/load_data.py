"""Functions for loading Mario replay data and metadata."""

import os
import os.path as op
import pandas as pd
import json
import logging


def _load_json_file(filepath):
    """Load and parse a single JSON file."""
    with open(filepath) as f:
        return json.load(f)


def _should_include_json(file, root, data_type):
    """Check if JSON file should be included based on type."""
    if not file.endswith(".json"):
        return False
    if data_type == 'metadata' and "infos" in root:
        return True
    if data_type == 'variables' and "variables" in root:
        return True
    return False


def _collect_json_files(replay_dir, data_type):
    """Collect all JSON files of specified type from directory."""
    sidecars_data = []
    for root, _, files in sorted(os.walk(replay_dir)):
        for file in files:
            if _should_include_json(file, root, data_type):
                filepath = op.join(root, file)
                sidecars_data.append(_load_json_file(filepath))
    return sidecars_data


def load_replay_data(replay_dir, type='metadata'):
    """
    Load sidecar files for replays in the specified directory.

    Args:
        replay_dir: Directory containing replay files and sidecars
        type: Type of data to load ('metadata' or 'variables')

    Returns:
        DataFrame with rows as individual replays and columns as sidecar data
    """
    sidecars_data = _collect_json_files(replay_dir, type)
    return pd.DataFrame(sidecars_data)


def _parse_entities_from_filename(basename):
    """Parse BIDS entities from filename."""
    entities = basename.split('_')
    entities_dict = {}
    for ent in entities:
        if '-' in ent:
            key, value = ent.split('-', 1)
            entities_dict[key] = value
    return entities_dict


def _extract_run_entities(run_events_file):
    """Extract subject, session, and run from events file name."""
    basename = op.basename(run_events_file)
    entities_dict = _parse_entities_from_filename(basename)
    return (
        entities_dict.get('sub'),
        entities_dict.get('ses'),
        entities_dict.get('run')
    )


def _should_skip_subject_session(sub, ses, subjects, sessions):
    """Check if subject/session should be filtered out."""
    if subjects and sub not in subjects:
        return True
    if sessions and ses not in sessions:
        return True
    return False


def _is_valid_bk2_file(bk2_file):
    """Check if bk2 file entry is valid."""
    return bk2_file != "Missing file" and not isinstance(bk2_file, float)


def _process_events_file(run_events_file, subjects, sessions):
    """Process a single events file and extract bk2 info."""
    logging.info(f"Processing events file: {op.basename(run_events_file)}")
    events_df = pd.read_table(run_events_file)
    events_df = events_df[events_df['trial_type'] == 'gym-retro_game']

    sub, ses, run = _extract_run_entities(run_events_file)

    if not all([sub, ses, run]):
        logging.warning(f"Missing entities in {op.basename(run_events_file)}")
        return []

    if _should_skip_subject_session(sub, ses, subjects, sessions):
        return []

    bk2_files_info = []
    bk2_files = events_df['stim_file'].values.tolist()
    for bk2_idx, bk2_file in enumerate(bk2_files):
        if _is_valid_bk2_file(bk2_file):
            bk2_files_info.append({
                'bk2_file': bk2_file,
                'bk2_idx': bk2_idx,
                'sub': sub,
                'ses': ses,
                'run': run
            })
    return bk2_files_info


def collect_bk2_files(DATA_PATH, subjects=None, sessions=None):
    """
    Collect all bk2 files and metadata from dataset.

    Args:
        DATA_PATH: Root path of the Mario dataset
        subjects: List of subject IDs to include (None for all)
        sessions: List of session IDs to include (None for all)

    Returns:
        List of dictionaries with bk2 file information
    """
    bk2_files_info = []
    for root, _, files in sorted(os.walk(DATA_PATH)):
        if "sourcedata" in root:
            continue

        for file in files:
            if "events.tsv" in file and "annotated" not in file:
                run_events_file = op.join(root, file)
                file_info = _process_events_file(run_events_file, subjects, sessions)
                bk2_files_info.extend(file_info)

    return bk2_files_info