 #------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
import re
import sys
import json
from pathlib import Path

import numpy as np
import torch
import wandb

from evaluation_helper import EvalHelper
from tabpfn.scripts.epoch_callback import epoch_callback
from tabpfn.scripts.model_builder import save_model
from tabpfn.scripts.model_builder_custom import get_model
from tabpfn.scripts.model_configs import *

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

print(f"currently in {os.getcwd()}")

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                        PARAMETER
#------------------------------------------------------------------------------------------------

base_path = "."
max_features = 100
json_file_path = "tabpfn_original_config.json"

#------------------------------------------------------------------------------------------------
#                                      END PARAMETER
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                         CONFIG
#------------------------------------------------------------------------------------------------

with open(json_file_path, "r") as f:
    config = json.load(f)

# Fill in stuff that could not be loaded properly into the config json.
uniform_int_sampler_f = lambda a, b: lambda: round(np.random.uniform(a, b))
choice_values = [
    torch.nn.modules.activation.Tanh,
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU,
]

config["differentiable_hyperparameters"]["prior_mlp_activations"]["choice_values"] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config["max_num_classes"])
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

#------------------------------------------------------------------------------------------------
#                                        END CONFIG
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                          CUSTOM
#------------------------------------------------------------------------------------------------

model_type = "hybrid"

config["batch_size"] = 64
config["emsize"] = 512
config["epochs"] = 200
config["bptt"] = 1024
config["max_eval_pos"] = 1000
config["num_steps"] = 584
config["hybrid_layer_types"] = (
    ["transformer"] * 2
    + ["hydra"] * 4
    + ["transformer"] * 2
)
config["nlayers"] = len(config["hybrid_layer_types"])
config["lr"] = 1e-4
lr_name = f"{config['lr']:g}".replace(".", "p")
checkpoint_model_name = f"tabpfn2_hydra4_tabpfn2_hybrid_8_layers_512e_lr{lr_name}"
config["enable_autocast"] = False
config["train_mixed_precision"] = False
config["enable_transformer_full_attn"] = False
config["bootstrap_samples"] = config["bptt"]
config["permutation_repeat"] = 0

# Keep the stability guards from the current custom run.
stability_overrides = config["differentiable_hyperparameters"]
stability_overrides["prior_mlp_dropout_prob"].update({
    "scale": 0.6,
    "min": 0.0,
    "max": 1.0,
})
stability_overrides["init_std"].update({
    "max_mean": 1.5,
})
stability_overrides["noise_std"].update({
    "max_mean": 0.1,
})
config["prior_mlp_max_abs_value"] = 1e4

device = os.environ.get("TRAIN_DEVICE", "cuda:2")
ENABLE_DATA_PARALLEL = False
CHECKPOINT_DIR = Path("tabpfn/models_diff")
RESUME_FROM_LATEST_CHECKPOINT = os.environ.get("RESUME_FROM_LATEST_CHECKPOINT", "1").lower() not in {
    "0",
    "false",
    "no",
}


def find_latest_callback_checkpoint(checkpoint_name):
    candidates = []
    pattern = re.compile(rf"callback_{re.escape(checkpoint_name)}_epoch_(\d+)\.cpkt$")
    for checkpoint_path in CHECKPOINT_DIR.glob(f"callback_{checkpoint_name}_epoch_*.cpkt"):
        match = pattern.match(checkpoint_path.name)
        if match:
            candidates.append((int(match.group(1)), checkpoint_path))

    latest_checkpoint = CHECKPOINT_DIR / f"callback_{checkpoint_name}_latest.cpkt"
    if latest_checkpoint.is_file():
        try:
            _, _, latest_config = torch.load(latest_checkpoint, map_location="cpu")
            candidates.append((int(latest_config.get("stop_epoch", 0)), latest_checkpoint))
        except Exception as exc:
            print(f"Could not inspect latest checkpoint {latest_checkpoint}: {exc}", flush=True)
    return max(candidates, default=(0, None))


def infer_checkpoint_epoch(checkpoint_path):
    match = re.search(r"_epoch_(\d+)\.cpkt$", checkpoint_path.name)
    return int(match.group(1)) if match else 0


def load_resume_checkpoint(checkpoint_path):
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}", flush=True)
    state_dict, _, checkpoint_config = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }
    checkpoint_epoch = int(
        checkpoint_config.get("stop_epoch") or infer_checkpoint_epoch(checkpoint_path)
    )
    return state_dict, checkpoint_epoch


resume_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT")
resume_start_epoch = 0
resume_state_dict = None

if resume_checkpoint is None and RESUME_FROM_LATEST_CHECKPOINT:
    resume_start_epoch, latest_checkpoint = find_latest_callback_checkpoint(checkpoint_model_name)
    resume_checkpoint = str(latest_checkpoint) if latest_checkpoint is not None else None

if resume_checkpoint:
    checkpoint_path = Path(resume_checkpoint)
    resume_state_dict, resume_start_epoch = load_resume_checkpoint(checkpoint_path)
    config["resume_checkpoint"] = str(checkpoint_path)
    config["resume_start_epoch"] = resume_start_epoch
    if resume_start_epoch >= config["epochs"]:
        print(
            f"Checkpoint is already at epoch {resume_start_epoch}, "
            f"which is >= configured epochs {config['epochs']}.",
            flush=True,
        )
    else:
        print(
            f"Resuming from epoch {resume_start_epoch}; next epoch is {resume_start_epoch + 1}.",
            flush=True,
        )
else:
    config["resume_checkpoint"] = None
    config["resume_start_epoch"] = 0
    if RESUME_FROM_LATEST_CHECKPOINT:
        print("No callback checkpoint found; starting from scratch.", flush=True)
    else:
        print("Checkpoint auto-resume disabled; starting from scratch.", flush=True)

#------------------------------------------------------------------------------------------------
#                                        END CUSTOM
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           WANDB
#------------------------------------------------------------------------------------------------

wandb_project = "mamba_project"
wandb_job_type = f"create_{model_type}_model"
wandb_run_name = (
    f"{model_type} tabpfn2-hydra4-tabpfn2 {config['nlayers']}l {config['emsize']}e "
    f"{config['batch_size']}b lr{config['lr']}_fp32"
)

wandb_mode = os.environ.get("WANDB_MODE")
if wandb_mode is None and not os.environ.get("WANDB_API_KEY"):
    wandb_mode = "offline"
    print("W&B API key is not configured; starting this run in offline mode.", flush=True)

wandb_run = wandb.init(
    project=wandb_project,
    job_type=wandb_job_type,
    config=config,
    name=wandb_run_name,
    group="DDP",
    mode=wandb_mode,
)

run_url = wandb_run.url or wandb_run.dir
print(f"WANDB_RUN_URL={run_url}", flush=True)
Path("wandb_run_url.txt").write_text(f"{run_url}\n")

#------------------------------------------------------------------------------------------------
#                                         END WANDB
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           MODEL
#------------------------------------------------------------------------------------------------

eval_class = EvalHelper()

hybrid_model = get_model(
    config,
    device,
    should_train=True,
    verbose=1,
    state_dict=resume_state_dict,
    epoch_callback=lambda model, epoch, config, _: epoch_callback(
        model,
        epoch,
        config,
        checkpoint_model_name,
    ),
    use_autocast=config["enable_autocast"],
    evaluation_class=eval_class,
    permutation_repeat=config["permutation_repeat"],
    bootstrap_samples=config["bootstrap_samples"],
    enable_data_parallel=ENABLE_DATA_PARALLEL,
    model_type=model_type,
)

(hp_embedding, data, _), targets, single_eval_pos = next(iter(hybrid_model[3]))

save_model(
    hybrid_model[2],
    base_path,
    f"tabpfn/models_diff/{checkpoint_model_name}_{config['nlayers']}l.cpkt",
    config,
)

#------------------------------------------------------------------------------------------------
#                                         END MODEL
#------------------------------------------------------------------------------------------------

wandb_run.finish()

print("works")
