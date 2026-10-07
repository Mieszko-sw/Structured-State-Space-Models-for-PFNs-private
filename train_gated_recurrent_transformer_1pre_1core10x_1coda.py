"""Train a three-physical-layer gated recurrent TabPFN.

Layout: one prelude layer, one shared middle layer applied up to ten times,
and one coda layer (1 + 1x10 + 1 = 12 effective layer executions).

During training, one depth is sampled uniformly from 1..10 for each forward
pass. Evaluation uses all ten recurrent steps unless num_loops is passed.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

from evaluation_helper import EvalHelper
from tabpfn.priors.utils import uniform_int_sampler_f
from tabpfn.scripts.epoch_callback import epoch_callback
from tabpfn.scripts.model_builder import save_model
from tabpfn.scripts.model_builder_custom import get_model
from tabpfn.scripts.model_configs import *  # noqa: F403


LOG_FILE_PATH = "gated_recurrent_transformer_1pre_1core10x_1coda.log"
CHECKPOINT_DIR = Path("tabpfn/models_diff")
CHECKPOINT_MODEL_NAME = "gated_recurrent_transformer_1pre_1core10x_1coda"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def checkpoint_epoch(checkpoint_path):
    match = re.search(r"_epoch_(\d+)\.cpkt$", checkpoint_path.name)
    return int(match.group(1)) if match else 0


def find_latest_checkpoint():
    candidates = []
    pattern = re.compile(
        rf"callback_{re.escape(CHECKPOINT_MODEL_NAME)}_epoch_(\d+)\.cpkt$"
    )
    for checkpoint_path in CHECKPOINT_DIR.glob(
        f"callback_{CHECKPOINT_MODEL_NAME}_epoch_*.cpkt"
    ):
        match = pattern.match(checkpoint_path.name)
        if match:
            candidates.append((int(match.group(1)), checkpoint_path))

    latest = CHECKPOINT_DIR / f"callback_{CHECKPOINT_MODEL_NAME}_latest.cpkt"
    if latest.is_file():
        try:
            _, _, latest_config = torch.load(latest, map_location="cpu")
            candidates.append((int(latest_config.get("stop_epoch", 0)), latest))
        except Exception as exc:
            print(f"Could not inspect latest checkpoint {latest}: {exc}", flush=True)
    return max(candidates, default=(0, None))


def load_resume_checkpoint(checkpoint_path):
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}", flush=True)
    state_dict, _, checkpoint_config = torch.load(
        checkpoint_path, map_location="cpu"
    )
    state_dict = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    epoch = int(
        checkpoint_config.get("stop_epoch") or checkpoint_epoch(checkpoint_path)
    )
    return state_dict, epoch


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

log_file = open(LOG_FILE_PATH, "a", buffering=1)
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)
print(f"currently in {os.getcwd()}")

with open("tabpfn_original_config.json", "r") as config_file:
    config = json.load(config_file)

max_features = 100
choice_values = [
    torch.nn.modules.activation.Tanh,
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU,
]
config["differentiable_hyperparameters"]["prior_mlp_activations"][
    "choice_values"
] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config["max_num_classes"])
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

model_type = "looped_transformer"
config["batch_size"] = 64
config["emsize"] = 512
config["epochs"] = 200
config["bptt"] = 1024
config["max_eval_pos"] = 1000
config["num_steps"] = 584

# Three physical layers and twelve maximum effective executions.
config["nlayers"] = 12
config["gated_recurrent_transformer"] = True
config["gated_warmup_layers"] = 1
config["gated_core_layers"] = 1
config["gated_exit_layers"] = 1
config["gated_max_loops"] = 10

# Paper-style uniform depth sampling and recurrent transition defaults.
config["gated_min_loops"] = 1
config["gated_sample_depth_during_training"] = True
config["gated_state_noise_std"] = 0.1
config["gated_gate_noise_std"] = 0.1
config["gated_gate_temperature"] = 1.0
config["gated_gate_bias"] = 4.0
config["elastic_shortcut_training"] = False
config["loopformer_shortcut_training"] = False

config["lr"] = 1e-4
config["enable_autocast"] = False
config["train_mixed_precision"] = False
config["enable_transformer_full_attn"] = False
config["bootstrap_samples"] = config["bptt"]
config["permutation_repeat"] = 0

# Match the stability overrides used by the other recurrent TabPFN runs.
stability_overrides = config["differentiable_hyperparameters"]
stability_overrides["prior_mlp_dropout_prob"].update(
    {"scale": 0.6, "min": 0.0, "max": 1.0}
)
stability_overrides["init_std"].update({"max_mean": 1.5})
stability_overrides["noise_std"].update({"max_mean": 0.1})
config["prior_mlp_max_abs_value"] = 1e4

device = os.environ.get("TRAIN_DEVICE", "cuda:6")
enable_data_parallel = False
auto_resume = os.environ.get("RESUME_FROM_LATEST_CHECKPOINT", "1").lower() not in {
    "0",
    "false",
    "no",
}

resume_path = os.environ.get("RESUME_FROM_CHECKPOINT")
resume_epoch = 0
resume_state_dict = None
if resume_path is None and auto_resume:
    resume_epoch, latest_path = find_latest_checkpoint()
    resume_path = str(latest_path) if latest_path is not None else None

if resume_path:
    resume_path = Path(resume_path)
    resume_state_dict, resume_epoch = load_resume_checkpoint(resume_path)
    config["resume_checkpoint"] = str(resume_path)
    config["resume_start_epoch"] = resume_epoch
    print(
        f"Resuming from epoch {resume_epoch}; next epoch is {resume_epoch + 1}.",
        flush=True,
    )
else:
    config["resume_checkpoint"] = None
    config["resume_start_epoch"] = 0
    print("No callback checkpoint found; starting from scratch.", flush=True)

wandb_mode = os.environ.get("WANDB_MODE")
if wandb_mode is None and not os.environ.get("WANDB_API_KEY"):
    wandb_mode = "offline"
    print("W&B API key is not configured; starting in offline mode.", flush=True)

wandb_run = wandb.init(
    project="mamba_project",
    job_type=f"create_{model_type}_model",
    config=config,
    name=(
        f"{CHECKPOINT_MODEL_NAME} {config['nlayers']}l {config['emsize']}e "
        f"{config['batch_size']}b lr{config['lr']}_fp32"
    ),
    group="DDP",
    mode=wandb_mode,
)
run_url = wandb_run.url or wandb_run.dir
print(f"WANDB_RUN_URL={run_url}", flush=True)
Path("gated_recurrent_wandb_run_url.txt").write_text(f"{run_url}\n")

evaluation = EvalHelper()
trained_model = get_model(
    config,
    device,
    should_train=True,
    verbose=1,
    state_dict=resume_state_dict,
    epoch_callback=lambda model, epoch, callback_config, _: epoch_callback(
        model, epoch, callback_config, CHECKPOINT_MODEL_NAME
    ),
    use_autocast=config["enable_autocast"],
    evaluation_class=evaluation,
    permutation_repeat=config["permutation_repeat"],
    bootstrap_samples=config["bootstrap_samples"],
    enable_data_parallel=enable_data_parallel,
    model_type=model_type,
)

# Materialize one batch, matching the existing training entry points.
next(iter(trained_model[3]))

save_model(
    trained_model[2],
    ".",
    f"tabpfn/models_diff/{CHECKPOINT_MODEL_NAME}_{config['nlayers']}l.cpkt",
    config,
)
wandb_run.finish()
print("works")

