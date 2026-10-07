"""Reproducible seed-10 launcher for elastic looped Transformer 3x6 training.

This intentionally writes to seed-specific logs and checkpoints so it does not
overwrite the unseeded or seed-0 runs. It starts from scratch by default.
"""

import os
import runpy
import sys
from pathlib import Path


SEED = "10"

# PYTHONHASHSEED is read only when Python starts, so restart this script once
# with the desired value when necessary.
if os.environ.get("PYTHONHASHSEED") != SEED:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = SEED
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )

os.environ["TRAIN_SEED"] = SEED
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault(
    "TRAIN_LOG_FILE_PATH",
    "elastic_looped_transformer_3physical_core6x_seed10.log",
)
os.environ.setdefault(
    "CHECKPOINT_MODEL_NAME",
    "elastic_looped_transformer_3physical_core6x_seed10",
)
os.environ.setdefault("RESUME_FROM_LATEST_CHECKPOINT", "0")
os.environ.setdefault(
    "WANDB_RUN_URL_FILE", "wandb_run_url_elastic_looped_3physical_core6x_seed10.txt"
)

runpy.run_path(
    str(Path(__file__).with_name("train_elastic_looped_transformer_3physical_core6x.py")),
    run_name="__main__",
)
