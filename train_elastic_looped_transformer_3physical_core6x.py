"""Train an elastic three-physical-layer TabPFN with up to six core-stack loops.

The core executes cyclically as (A -> B -> C)^d. Each update uses the full
d=6 trajectory (18 block applications) and one uniformly sampled shortcut
d in {1, 2, 3, 4, 5} (3, 6, 9, 12, or 15 block applications). The loss combines full CE,
shortcut CE, and detached full-to-short temperature-scaled KL distillation. This is the shortcut-distillation baseline;
it does not yet add LoopFormer time/step-size modulation inside the block.
"""

#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
import sys
import re
import random

TRAIN_SEED_ENV = os.environ.get("TRAIN_SEED")
if TRAIN_SEED_ENV is not None:
    # Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.
    # This must be set before importing torch.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LOG_FILE_PATH = os.environ.get(
    "TRAIN_LOG_FILE_PATH",
    "elastic_looped_transformer_3physical_core6x.log",
)

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

log_file = open(LOG_FILE_PATH, "a", buffering=1)
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"currently in {os.getcwd()}")
from datetime import datetime

import torch
# Limit PyTorch CUDA use
#torch.cuda.set_per_process_memory_fraction(0.6)
import json

from pathlib import Path

import numpy as np
import wandb
from tabpfn.scripts.model_builder import save_model
from tabpfn.scripts.model_builder_custom import get_model
from tabpfn.scripts.model_configs import *
from tabpfn.scripts.epoch_callback import epoch_callback
from tabpfn.priors.utils import uniform_int_sampler_f

from evaluation_helper import EvalHelper

if TRAIN_SEED_ENV is not None:
    TRAIN_SEED = int(TRAIN_SEED_ENV)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print(f"Using fixed training seed {TRAIN_SEED} with deterministic algorithms.")
else:
    TRAIN_SEED = None

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                        PARAMETER
#------------------------------------------------------------------------------------------------

# Other Parameters
base_path = '.'
max_features = 100
large_datasets = True

# Others
json_file_path = "tabpfn_original_config.json"

#------------------------------------------------------------------------------------------------
#                                      END PARAMETER
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                         CONFIG
#------------------------------------------------------------------------------------------------

with open(json_file_path, "r") as f: config = json.load(f)

if TRAIN_SEED is not None:
    config["seed"] = TRAIN_SEED
    
# Fill in stuff that could not be loaded properly into the config.json
uniform_int_sampler_f = (lambda a, b : lambda : round(np.random.uniform(a, b)))
choice_values = [
    torch.nn.modules.activation.Tanh, 
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU
    ]

config["differentiable_hyperparameters"]["prior_mlp_activations"]["choice_values"] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config['max_num_classes']) # Wrong Function
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

#------------------------------------------------------------------------------------------------
#                                        END CONFIG
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                          CUSTOM
#------------------------------------------------------------------------------------------------

model_type = "looped_transformer"
checkpoint_model_name = os.environ.get(
    "CHECKPOINT_MODEL_NAME",
    "elastic_looped_transformer_3physical_core6x",
)

config['batch_size'] = 64 
config['emsize'] = 512 
config["epochs"] = 200
config["bptt"] = 1024
config["max_eval_pos"] = 1000        

config["num_steps"] = 584

config["nlayers"] = 18
config["looped_warmup_layers"] = 0
config["looped_core_layers"] = 3
config["looped_exit_layers"] = 0
config["looped_core_repeat_pattern"] = None

# Elastic shortcut-distillation objective. One shared core-stack depth is sampled per batch.
config["elastic_shortcut_training"] = True
config["elastic_max_loops"] = 6
config["elastic_shortcut_min_loops"] = 1
config["elastic_shortcut_ce_weight"] = 0.1
config["elastic_consistency_weight"] = 0.1
config["elastic_distillation_temperature"] = 2.0
config["lr"] = 1e-4
config["enable_autocast"] = False
config["train_mixed_precision"] = False
config["enable_transformer_full_attn"] = False
config["bootstrap_samples"] = config["bptt"]          # Default would be 0.
config["permutation_repeat"] = 0

# Conservative stability settings for the recurrent Transformer run.  The original MLP prior
# ranges can create extreme finite values that later explode into NaN/Inf
# gradients; keep the same prior family, but narrow the most volatile tails.
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

device = os.environ.get("TRAIN_DEVICE", "cuda:1")
ENABLE_DATA_PARALLEL = False
CHECKPOINT_DIR = Path("tabpfn/models_diff")
RESUME_FROM_LATEST_CHECKPOINT = os.environ.get("RESUME_FROM_LATEST_CHECKPOINT", "1").lower() not in {
    "0",
    "false",
    "no",
}

def find_latest_callback_checkpoint(checkpoint_model_name):
    candidates = []
    pattern = re.compile(rf"callback_{re.escape(checkpoint_model_name)}_epoch_(\d+)\.cpkt$")
    for checkpoint_path in CHECKPOINT_DIR.glob(f"callback_{checkpoint_model_name}_epoch_*.cpkt"):
        match = pattern.match(checkpoint_path.name)
        if match:
            candidates.append((int(match.group(1)), checkpoint_path))
    latest_checkpoint = CHECKPOINT_DIR / f"callback_{checkpoint_model_name}_latest.cpkt"
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, _, checkpoint_config = checkpoint
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }
    checkpoint_epoch = int(
        checkpoint_config.get("stop_epoch") or infer_checkpoint_epoch(checkpoint_path)
    )
    return state_dict, checkpoint_config, checkpoint_epoch

resume_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT")
resume_start_epoch = 0
resume_state_dict = None

if resume_checkpoint is None and RESUME_FROM_LATEST_CHECKPOINT:
    resume_start_epoch, latest_checkpoint = find_latest_callback_checkpoint(checkpoint_model_name)
    resume_checkpoint = str(latest_checkpoint) if latest_checkpoint is not None else None

if resume_checkpoint:
    checkpoint_path = Path(resume_checkpoint)
    resume_state_dict, checkpoint_config, checkpoint_epoch = load_resume_checkpoint(checkpoint_path)
    resume_start_epoch = checkpoint_epoch
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

#os.environ["SLURM_PROCID"]="1"

#------------------------------------------------------------------------------------------------
#                                        END CUSTOM
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           WANDB
#------------------------------------------------------------------------------------------------

wandb_project = "mamba_project"
wandb_job_type = f"create_{model_type}_model"
wandb_run_name = f"{checkpoint_model_name} {config['nlayers']}l {config['emsize']}e {config['batch_size']}b lr{config['lr']}_fp32"

wandb_config= config

wandb_mode = os.environ.get("WANDB_MODE")
if wandb_mode is None and not os.environ.get("WANDB_API_KEY"):
    wandb_mode = "offline"
    print("W&B API key is not configured; starting this run in offline mode.", flush=True)

wandb_run = wandb.init(
    project=wandb_project,
    job_type=wandb_job_type,
    config=wandb_config,
    name=wandb_run_name,
    group="DDP",
    mode=wandb_mode,
)

run_url = wandb_run.url or wandb_run.dir
print(f"WANDB_RUN_URL={run_url}", flush=True)
Path(
    os.environ.get("WANDB_RUN_URL_FILE", "wandb_run_url.txt")
).write_text(f"{run_url}\n")

#------------------------------------------------------------------------------------------------
#                                         END WANDB
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           MODEL
#------------------------------------------------------------------------------------------------

# Evaluation during training:
eval_class = EvalHelper()

# Get the model 
#model = get_model(config, device, should_train=True, verbose=0) # , state_dict=model[2].state_dict()
looped_model = get_model(config, 
                              device, 
                              should_train=True, 
                              verbose=1,
                              state_dict=resume_state_dict,
                              epoch_callback=lambda model, epoch, config, _: epoch_callback(model, epoch, config, checkpoint_model_name),
                              use_autocast=config["enable_autocast"], 
                              evaluation_class=eval_class,
                              permutation_repeat=config["permutation_repeat"],
                              bootstrap_samples = config["bootstrap_samples"],
                              enable_data_parallel=ENABLE_DATA_PARALLEL,
                              model_type=model_type
                              ) # , state_dict=model[2].state_dict()

(hp_embedding, data, _), targets, single_eval_pos = next(iter(looped_model[3]))

# Save the elastic three-physical-layer, maximum-six-stack-loop TabPFN model
save_model(looped_model[2], 
           base_path, 
           f"tabpfn/models_diff/{checkpoint_model_name}_{config['nlayers']}l.cpkt",
           config
           )

#------------------------------------------------------------------------------------------------
#                                         END MODEL
#------------------------------------------------------------------------------------------------


wandb_run.finish()

print("works")
