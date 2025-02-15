from invoke import task
import os.path as op

BASE_DIR = op.dirname(op.abspath(__file__))

# ===============================
# 🔹 TASKS: Data Processing
# ===============================

@task
def generate_files(c):
    """Generates files."""
    c.run(f"python {BASE_DIR}/src/mario_replays/generate_files/generate_files.py -d data/mario")

@task
def set_mario_dataset(c):
    """Sets up the Mario dataset."""
    command = (
        "mkdir -p data && "
        "cd data && "
        "datalad install git@github.com:courtois-neuromod/mario && "
        "cd mario && "
        "git checkout events && "
        "datalad get */*/*/*.bk2 && "
        "datalad get */*/*/*.tsv &&"
        "rm -rf stimuli && "
        "datalad install git@github.com:courtois-neuromod/mario.stimuli stimuli && "
        "cd stimuli && "
        "git checkout scenes_states && "
        "datalad get ."
    )
    c.run(command)


# ===============================
# 🔹 TASKS: Utility & Maintenance
# ===============================

@task
def setup_env(c):
    """Sets up the virtual environment and installs dependencies."""
    c.run("pip install -r requirements.txt")
    c.run("pip install -e .")