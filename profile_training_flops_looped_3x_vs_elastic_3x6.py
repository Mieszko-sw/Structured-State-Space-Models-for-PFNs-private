"""Compare training FLOPs of fixed-depth looped 3xD models and elastic 3x6.

The default comparison matches the completed training runs in this repository:

* fixed looped 3x2, 3x3, 3x5, and 3x6;
* elastic looped 3x6, trained with a full depth-6 trajectory and one shortcut
  trajectory sampled uniformly from depths 1, 2, 3, 4, and 5.

Without ``--profile``, the script reports architecture-normalized compute in
core-loop and Transformer-block applications.  This comparison is exact for
the recurrent backbone but excludes the relatively small encoders, decoder,
loss, and optimizer-update overheads.

With ``--profile``, a real forward+loss+backward training microbatch is measured
with ``torch.profiler``.  Every elastic shortcut depth is measured separately
and combined with its exact uniform probability.  The profiler only assigns
FLOPs to supported PyTorch operators, so these values are deliberately named
``profiler_counted_flops``.  Optimizer updates and synthetic-data generation
are excluded from both sides.

Examples
--------
Architecture-normalized comparison (fast, CPU-only)::

    python3 profile_training_flops_looped_3x_vs_elastic_3x6.py

Measured training-step comparison on the actual batch shape::

    python3 profile_training_flops_looped_3x_vs_elastic_3x6.py \
        --profile --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from tabpfn.scripts.model_builder_custom import load_model_only_inference


CORE_LAYERS = 3
DEFAULT_FIXED_DEPTHS = (2, 3, 5, 6)
DEFAULT_SHORTCUT_DEPTHS = (1, 2, 3, 4, 5)
DEFAULT_MAX_DEPTH = 6
DEFAULT_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_3physical_core6x_latest.cpkt"
)
DEFAULT_CSV = os.path.join(
    "result_csvs", "training_flops_looped_3x_vs_elastic_3x6.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-depths",
        nargs="+",
        type=int,
        default=list(DEFAULT_FIXED_DEPTHS),
        help="Fixed model depths to compare (default: 2 3 5 6).",
    )
    parser.add_argument(
        "--elastic-max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Full elastic trajectory depth (default: 6).",
    )
    parser.add_argument(
        "--shortcut-depths",
        nargs="+",
        type=int,
        default=list(DEFAULT_SHORTCUT_DEPTHS),
        help="Uniformly sampled elastic shortcut depths (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. In profile mode, defaults to the checkpoint value; otherwise 200.",
    )
    parser.add_argument(
        "--microsteps-per-epoch",
        type=int,
        default=None,
        help=(
            "Forward/backward microsteps per epoch. In profile mode, defaults to the "
            "checkpoint's num_steps (4672 for these runs); otherwise 4672."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Measure supported forward+loss+backward FLOPs with torch.profiler.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Synthetic sequence length (default: checkpoint bptt, normally 1024).",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Synthetic microbatch size (default: checkpoint batch_size, normally 8).",
    )
    parser.add_argument(
        "--eval-position",
        type=int,
        default=512,
        help="Context/query boundary in the synthetic sequence (default: 512).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Unprofiled training steps per operating point (default: 1).",
    )
    parser.add_argument(
        "--shortcut-ce-weight", type=float, default=0.1
    )
    parser.add_argument(
        "--consistency-weight", type=float, default=0.1
    )
    parser.add_argument(
        "--distillation-temperature", type=float, default=2.0
    )
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    return parser.parse_args()


def unique_positive(values: Iterable[int], label: str) -> List[int]:
    result = list(dict.fromkeys(values))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{label} must contain positive integer depths.")
    return result


def validate_args(args: argparse.Namespace) -> None:
    args.fixed_depths = unique_positive(args.fixed_depths, "--fixed-depths")
    args.shortcut_depths = unique_positive(args.shortcut_depths, "--shortcut-depths")
    if args.elastic_max_depth <= 0:
        raise ValueError("--elastic-max-depth must be positive.")
    if any(depth >= args.elastic_max_depth for depth in args.shortcut_depths):
        raise ValueError("Every shortcut depth must be below --elastic-max-depth.")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.microsteps_per_epoch is not None and args.microsteps_per_epoch <= 0:
        raise ValueError("--microsteps-per-epoch must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.distillation_temperature <= 0:
        raise ValueError("--distillation-temperature must be positive.")


def normalize_device(device: str) -> str:
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        torch.cuda.set_device(torch.device(device))
    return device


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def make_synthetic_batch(
    config: dict,
    device: str,
    sequence_length: int,
    microbatch_size: int,
    eval_position: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if not 0 < eval_position < sequence_length:
        raise ValueError("--eval-position must lie strictly inside the sequence.")

    num_features = int(config["num_features"])
    num_classes = int(config["max_num_classes"])
    generator = torch.Generator(device=device).manual_seed(13)
    x = torch.randn(
        sequence_length,
        microbatch_size,
        num_features,
        generator=generator,
        device=device,
    )
    y = torch.randint(
        num_classes,
        (sequence_length, microbatch_size),
        generator=generator,
        device=device,
    ).float()
    targets = y[eval_position:].long()
    return (x, y), targets


def fixed_loss(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    eval_position: int,
    depth: int,
) -> torch.Tensor:
    output = model(inputs, single_eval_pos=eval_position, num_loops=depth)
    return F.cross_entropy(
        output.reshape(-1, output.shape[-1]), targets.reshape(-1)
    )


def elastic_loss(
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    eval_position: int,
    max_depth: int,
    shortcut_depth: int,
    shortcut_ce_weight: float,
    consistency_weight: float,
    temperature: float,
) -> torch.Tensor:
    full_output = model(
        inputs, single_eval_pos=eval_position, num_loops=max_depth
    )
    shortcut_output = model(
        inputs, single_eval_pos=eval_position, num_loops=shortcut_depth
    )

    full_ce = F.cross_entropy(
        full_output.reshape(-1, full_output.shape[-1]), targets.reshape(-1)
    )
    shortcut_ce = F.cross_entropy(
        shortcut_output.reshape(-1, shortcut_output.shape[-1]),
        targets.reshape(-1),
    )
    teacher_probs = torch.softmax(full_output.detach() / temperature, dim=-1)
    student_log_probs = torch.log_softmax(shortcut_output / temperature, dim=-1)
    consistency = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).sum(dim=-1).mean() * (temperature ** 2)
    return full_ce + shortcut_ce_weight * shortcut_ce + consistency_weight * consistency


def run_loss_and_backward(model: torch.nn.Module, loss_fn) -> None:
    model.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()


def profiler_activities(device: str) -> List[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profile_backward_step(
    model: torch.nn.Module,
    loss_fn,
    device: str,
    warmup_runs: int,
) -> Tuple[int, str]:
    for _ in range(warmup_runs):
        run_loss_and_backward(model, loss_fn)
    synchronize(device)

    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=profiler_activities(device),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
    ) as prof:
        loss = loss_fn()
        loss.backward()
        synchronize(device)

    counted_events = [
        event for event in prof.key_averages() if (event.flops or 0) > 0
    ]
    counted_flops = sum(int(event.flops) for event in counted_events)
    operator_types = ";".join(sorted(event.key for event in counted_events))
    return counted_flops, operator_types


def load_profile_model(args: argparse.Namespace):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    loaded, config = load_model_only_inference(
        ".", str(checkpoint), args.device, model_name="looped_transformer"
    )
    model = loaded[2]
    model.train()
    return model, config


def profile_operating_points(
    args: argparse.Namespace,
) -> Tuple[Dict[int, int], Dict[int, int], str, dict]:
    model, config = load_profile_model(args)
    sequence_length = args.sequence_length or int(config["bptt"])
    microbatch_size = args.microbatch_size or int(config["batch_size"])
    inputs, targets = make_synthetic_batch(
        config,
        args.device,
        sequence_length,
        microbatch_size,
        args.eval_position,
    )

    fixed_flops: Dict[int, int] = {}
    elastic_flops: Dict[int, int] = {}
    operator_types = set()

    for depth in args.fixed_depths:
        loss_fn = lambda depth=depth: fixed_loss(
            model, inputs, targets, args.eval_position, depth
        )
        flops, operators = profile_backward_step(
            model, loss_fn, args.device, args.warmup_runs
        )
        fixed_flops[depth] = flops
        operator_types.update(filter(None, operators.split(";")))
        print(
            f"profiled fixed 3x{depth}: {flops / 1e12:.6f} TFLOPs/microbatch",
            flush=True,
        )

    for shortcut_depth in args.shortcut_depths:
        loss_fn = lambda shortcut_depth=shortcut_depth: elastic_loss(
            model,
            inputs,
            targets,
            args.eval_position,
            args.elastic_max_depth,
            shortcut_depth,
            args.shortcut_ce_weight,
            args.consistency_weight,
            args.distillation_temperature,
        )
        flops, operators = profile_backward_step(
            model, loss_fn, args.device, args.warmup_runs
        )
        elastic_flops[shortcut_depth] = flops
        operator_types.update(filter(None, operators.split(";")))
        print(
            f"profiled elastic 3x6 shortcut={shortcut_depth}: "
            f"{flops / 1e12:.6f} TFLOPs/microbatch",
            flush=True,
        )

    metadata = {
        "sequence_length": sequence_length,
        "microbatch_size": microbatch_size,
        "aggregate_k_gradients": int(config.get("aggregate_k_gradients", 1)),
        "checkpoint_num_steps": int(config["num_steps"]),
        "checkpoint_epochs": int(config["epochs"]),
    }
    return fixed_flops, elastic_flops, ";".join(sorted(operator_types)), metadata


def build_rows(
    args: argparse.Namespace,
    epochs: int,
    microsteps_per_epoch: int,
    fixed_flops: Optional[Dict[int, int]] = None,
    elastic_flops: Optional[Dict[int, int]] = None,
    operator_types: str = "",
) -> List[dict]:
    total_microsteps = epochs * microsteps_per_epoch
    expected_shortcut_depth = sum(args.shortcut_depths) / len(args.shortcut_depths)
    elastic_expected_loops = args.elastic_max_depth + expected_shortcut_depth
    elastic_expected_blocks = CORE_LAYERS * elastic_expected_loops
    fixed_total_loops = float(sum(args.fixed_depths))
    fixed_total_blocks = CORE_LAYERS * fixed_total_loops

    expected_elastic_profiled = None
    if elastic_flops is not None:
        expected_elastic_profiled = sum(elastic_flops.values()) / len(elastic_flops)

    rows = []
    for depth in args.fixed_depths:
        per_microbatch = fixed_flops[depth] if fixed_flops is not None else None
        rows.append(
            {
                "model": f"looped_3x{depth}",
                "comparison_role": "individual_fixed_model",
                "full_depth": depth,
                "shortcut_depths": "",
                "expected_core_loops_per_microbatch": float(depth),
                "expected_transformer_blocks_per_microbatch": float(CORE_LAYERS * depth),
                "epochs": epochs,
                "microsteps_per_epoch": microsteps_per_epoch,
                "total_microsteps": total_microsteps,
                "profiler_counted_flops_per_microbatch": per_microbatch,
                "profiler_counted_total_training_flops": (
                    per_microbatch * total_microsteps if per_microbatch is not None else None
                ),
                "normalized_compute_ratio_vs_elastic": depth / elastic_expected_loops,
                "normalized_compute_difference_vs_elastic": depth - elastic_expected_loops,
                "profiler_ratio_vs_elastic": (
                    per_microbatch / expected_elastic_profiled
                    if per_microbatch is not None and expected_elastic_profiled
                    else None
                ),
                "counted_operator_types": operator_types,
            }
        )

    fixed_total_profiled = sum(fixed_flops.values()) if fixed_flops is not None else None
    rows.append(
        {
            "model": "fixed_models_total",
            "comparison_role": "sum_of_fixed_models",
            "full_depth": "",
            "shortcut_depths": "",
            "expected_core_loops_per_microbatch": fixed_total_loops,
            "expected_transformer_blocks_per_microbatch": fixed_total_blocks,
            "epochs": epochs,
            "microsteps_per_epoch": microsteps_per_epoch,
            "total_microsteps": total_microsteps,
            "profiler_counted_flops_per_microbatch": fixed_total_profiled,
            "profiler_counted_total_training_flops": (
                fixed_total_profiled * total_microsteps
                if fixed_total_profiled is not None
                else None
            ),
            "normalized_compute_ratio_vs_elastic": fixed_total_loops / elastic_expected_loops,
            "normalized_compute_difference_vs_elastic": fixed_total_loops - elastic_expected_loops,
            "profiler_ratio_vs_elastic": (
                fixed_total_profiled / expected_elastic_profiled
                if fixed_total_profiled is not None and expected_elastic_profiled
                else None
            ),
            "counted_operator_types": operator_types,
        }
    )
    rows.append(
        {
            "model": "elastic_looped_3x6_expected",
            "comparison_role": "elastic_model",
            "full_depth": args.elastic_max_depth,
            "shortcut_depths": ",".join(map(str, args.shortcut_depths)),
            "expected_core_loops_per_microbatch": elastic_expected_loops,
            "expected_transformer_blocks_per_microbatch": elastic_expected_blocks,
            "epochs": epochs,
            "microsteps_per_epoch": microsteps_per_epoch,
            "total_microsteps": total_microsteps,
            "profiler_counted_flops_per_microbatch": expected_elastic_profiled,
            "profiler_counted_total_training_flops": (
                expected_elastic_profiled * total_microsteps
                if expected_elastic_profiled is not None
                else None
            ),
            "normalized_compute_ratio_vs_elastic": 1.0,
            "normalized_compute_difference_vs_elastic": 0.0,
            "profiler_ratio_vs_elastic": 1.0 if expected_elastic_profiled is not None else None,
            "counted_operator_types": operator_types,
        }
    )
    return rows


def write_csv(rows: List[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[dict], profiled: bool) -> None:
    elastic = next(row for row in rows if row["comparison_role"] == "elastic_model")
    fixed_total = next(
        row for row in rows if row["comparison_role"] == "sum_of_fixed_models"
    )
    normalized_saving = 1.0 - (
        elastic["expected_core_loops_per_microbatch"]
        / fixed_total["expected_core_loops_per_microbatch"]
    )

    print("\nArchitecture-normalized comparison")
    print("model                         expected loops   expected block applications")
    for row in rows:
        print(
            f"{row['model']:<29}"
            f"{row['expected_core_loops_per_microbatch']:>14.3f}"
            f"{row['expected_transformer_blocks_per_microbatch']:>30.3f}"
        )
    print(
        f"\nElastic versus all fixed models: {normalized_saving * 100:.2f}% "
        "fewer recurrent-backbone FLOPs."
    )

    if profiled:
        elastic_total = elastic["profiler_counted_total_training_flops"]
        fixed_total_flops = fixed_total["profiler_counted_total_training_flops"]
        measured_saving = 1.0 - elastic_total / fixed_total_flops
        print("\nProfiler-counted forward+loss+backward comparison")
        print("model                         TFLOPs/microbatch   total training FLOPs")
        for row in rows:
            print(
                f"{row['model']:<29}"
                f"{row['profiler_counted_flops_per_microbatch'] / 1e12:>18.6f}"
                f"{row['profiler_counted_total_training_flops']:>23.6e}"
            )
        print(
            f"\nElastic versus all fixed models: {measured_saving * 100:.2f}% "
            "fewer profiler-counted training FLOPs."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    fixed_flops = None
    elastic_flops = None
    operator_types = ""
    metadata = {}
    if args.profile:
        args.device = normalize_device(args.device)
        fixed_flops, elastic_flops, operator_types, metadata = profile_operating_points(args)

    epochs = args.epochs or metadata.get("checkpoint_epochs", 200)
    microsteps_per_epoch = args.microsteps_per_epoch or metadata.get(
        "checkpoint_num_steps", 4672
    )
    rows = build_rows(
        args,
        epochs,
        microsteps_per_epoch,
        fixed_flops,
        elastic_flops,
        operator_types,
    )
    write_csv(rows, args.output_csv)
    print_summary(rows, args.profile)
    print(f"\nWrote {args.output_csv}")

    if metadata:
        print(
            "Profile shape: "
            f"sequence_length={metadata['sequence_length']}, "
            f"microbatch_size={metadata['microbatch_size']}, "
            f"gradient_accumulation={metadata['aggregate_k_gradients']}."
        )
    print(
        "Note: profiler counts cover supported operators in forward, loss, and backward; "
        "optimizer updates and data generation are excluded."
    )


if __name__ == "__main__":
    main()
