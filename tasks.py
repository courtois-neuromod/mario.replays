from invoke import task
import os.path as op

BASE_DIR = op.dirname(op.abspath(__file__))

# ===============================
# 🔹 TASKS: Data Processing
# ===============================

@task
def setup_mario_dataset(c):
    """Sets up the Mario dataset."""
    command = (
        "mkdir -p data && "
        "cd data && "
        "datalad install git@github.com:courtois-neuromod/mario && "
        "cd mario && "
        "git checkout events && "
        "datalad get */*/*/*.bk2 && "
        "datalad get */*/*/*.tsv &&"
        "rm -rf stimuli && " ### the following part is to be removed after mario.stimuli is updated
        "datalad install git@github.com:courtois-neuromod/mario.stimuli stimuli && "
        "cd stimuli && "
        "git checkout scenes_states && "
        "datalad get ."
    )
    c.run(command)

@task
def create_replays(c):
    """Generates files."""
    c.run(f"python {BASE_DIR}/src/mario_replays/create_replays/create_replays.py -d data/mario")



# ===============================
# 🔹 TASKS: Utility & Maintenance
# ===============================

@task
def setup_env(c):
    """Sets up the virtual environment and installs dependencies."""
    c.run("pip install -r requirements.txt")
    c.run("pip install -e .")

@task
def setup_env_compute_canada(c):
    """Sets up the virtual environment and installs dependencies."""
    c.run("git clone git@github.com:FaramaFoundation/stable-retro")
    c.run("cd stable-retro")
    c.run("pip install -e .")
    c.run("cd ..")
    c.run("pip install -r requirements.txt")
    c.run("pip install -e .")

@task
def full_pipeline(c):
    """Runs the full pipeline."""
    c.run("invoke setup-env")
    c.run("invoke setup-mario-dataset")
    c.run("invoke create-replays")