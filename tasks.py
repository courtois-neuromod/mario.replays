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
        f"source {BASE_DIR}/env/bin/activate && "
        "mkdir -p sourcedata && "
        "cd sourcedata && "
        "datalad install git@github.com:courtois-neuromod/mario && "# get stimuli through submodule #"datalad install -s ria+ssh://elm.criugm.qc.ca/data/neuromod/ria-sequoia#~cneuromod.mario.raw@events mario && "
        "cd mario && "
        "git checkout events && "
        "datalad get */*/*/*.bk2 && "
        "datalad get */*/*/*.tsv &&"
        "rm -rf stimuli && "
        "datalad install git@github.com:courtois-neuromod/mario.stimuli && "
        "mv mario.stimuli stimuli && "
        "cd stimuli && "
        "git checkout scenes_states && "
        "datalad get ."
    )
    c.run(command)

@task
def create_replays(c):
    """Generates files."""
    c.run(f"python {BASE_DIR}/code/mario_replays/create_replays/create_replays.py -d data/mario")



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