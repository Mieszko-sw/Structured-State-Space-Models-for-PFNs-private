"""Measure training-step GPU energy for fixed and elastic Looped3x models.

This is a training benchmark, not an inference benchmark.  Every measured
microbatch executes the forward pass, training losses, backward pass, gradient
clipping, and an AdamW update at the checkpoint's gradient-accumulation
frequency.  The elastic operating points execute the full maximum-depth path
and one shortcut path, including the shortcut CE and distillation losses used
during ElTab training.

The benchmark loads one completed elastic checkpoint because all compared
models share the same three-layer physical core.  Selecting ``num_loops``
reproduces the fixed-depth compute graphs while holding weights and inputs
constant.  Checkpoint values are never modified or saved.

GPU energy is read from NVML's cumulative millijoule counter.  The default
three repetitions, 512 measured microbatches per operating point, and balanced
shortcut depths are intended to finish comfortably within 12 hours on an A100.

Example
-------
Run on a dedicated physical GPU 2 and supply a location-specific grid factor::

    python3 measure_training_energy_looped_vs_elastic.py \
        --device cuda:2 --nvml-gpu-index 2 \
        --carbon-intensity-g-per-kwh 350 --pue 1.2

The percentage energy reduction does not depend on PUE or grid carbon
intensity when every operating point is measured under the same conditions.
Those inputs are needed only for absolute CO2e estimates.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from profile_training_flops_looped_3x_vs_elastic_3x6 import (
    elastic_loss,
    fixed_loss,
    make_synthetic_batch,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference


DEFAULT_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_3physical_core6x_latest.cpkt"
)
DEFAULT_RAW_CSV = "result_csvs/training_energy_looped_3x_vs_elastic_raw.csv"
DEFAULT_SUMMARY_CSV = (
    "result_csvs/training_energy_looped_3x_vs_elastic_summary.csv"
)
JOULES_PER_KWH = 3_600_000.0


@dataclass(frozen=True)
class OperatingPoint:
    model: str
    role: str
    fixed_depth: Optional[int] = None
    shortcut_depth: Optional[int] = None


@dataclass
class NvmlEnergyMeter:
    pynvml: object
    handle: object
    device_label: str

    def read_millijoules(self) -> int:
        return int(self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))

    def compute_process_ids(self) -> List[int]:
        for function_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            function = getattr(self.pynvml, function_name, None)
            if function is not None:
                return sorted({int(process.pid) for process in function(self.handle)})
        return []

    def metadata(self) -> dict:
        def decode(value):
            return value.decode() if isinstance(value, bytes) else str(value)

        return {
            "gpu_name": decode(self.pynvml.nvmlDeviceGetName(self.handle)),
            "gpu_uuid": decode(self.pynvml.nvmlDeviceGetUUID(self.handle)),
            "gpu_power_limit_watts": (
                self.pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
            ),
            "nvml_device": self.device_label,
        }

    def shutdown(self) -> None:
        self.pynvml.nvmlShutdown()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nvml-gpu-index",
        type=int,
        default=None,
        help=(
            "Physical NVML GPU index. By default it is inferred from --device "
            "and CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Allow measurement while other compute processes use the GPU (not recommended).",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--fixed-depths", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6]
    )
    parser.add_argument(
        "--shortcut-depths", nargs="+", type=int, default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--elastic-max-depth", type=int, default=6)
    parser.add_argument(
        "--warmup-microbatches",
        type=int,
        default=32,
        help="Unmeasured training microbatches before every operating point.",
    )
    parser.add_argument(
        "--measured-microbatches",
        type=int,
        default=512,
        help="Measured training microbatches per operating point and repetition.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Defaults to the checkpoint bptt value.",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Defaults to the checkpoint's effective microbatch size.",
    )
    parser.add_argument("--eval-position", type=int, default=512)
    parser.add_argument(
        "--aggregate-k-gradients",
        type=int,
        default=None,
        help="Defaults to the checkpoint's gradient-accumulation factor.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Defaults to the checkpoint value."
    )
    parser.add_argument(
        "--microsteps-per-epoch",
        type=int,
        default=None,
        help="Defaults to the checkpoint's effective num_steps value.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--shortcut-ce-weight", type=float, default=None)
    parser.add_argument("--consistency-weight", type=float, default=None)
    parser.add_argument("--distillation-temperature", type=float, default=None)
    parser.add_argument(
        "--carbon-intensity-g-per-kwh",
        type=float,
        default=None,
        help="Optional location/time-specific electricity carbon intensity.",
    )
    parser.add_argument(
        "--pue",
        type=float,
        default=1.0,
        help="Power usage effectiveness; 1.0 reports GPU-only operational energy.",
    )
    parser.add_argument("--raw-output-csv", default=DEFAULT_RAW_CSV)
    parser.add_argument("--summary-output-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--fixed-order",
        action="store_true",
        help="Do not shuffle operating-point order between repetitions.",
    )
    return parser.parse_args(argv)


def unique_positive(values: Iterable[int], name: str) -> List[int]:
    result = list(dict.fromkeys(values))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integer values.")
    return result


def validate_args(args: argparse.Namespace) -> None:
    args.fixed_depths = unique_positive(args.fixed_depths, "--fixed-depths")
    args.shortcut_depths = unique_positive(args.shortcut_depths, "--shortcut-depths")
    if args.elastic_max_depth <= 0:
        raise ValueError("--elastic-max-depth must be positive.")
    if any(depth >= args.elastic_max_depth for depth in args.shortcut_depths):
        raise ValueError("Every shortcut depth must be below --elastic-max-depth.")
    for name in ("warmup_microbatches", "measured_microbatches", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.nvml_gpu_index is not None and args.nvml_gpu_index < 0:
        raise ValueError("--nvml-gpu-index must be nonnegative.")
    if args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be positive.")
    if args.carbon_intensity_g_per_kwh is not None and args.carbon_intensity_g_per_kwh < 0:
        raise ValueError("--carbon-intensity-g-per-kwh must be nonnegative.")
    if args.pue < 1.0:
        raise ValueError("--pue must be at least 1.0.")


def normalize_device(device: str) -> str:
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device == "cuda":
        device = "cuda:0"
    if not device.startswith("cuda:"):
        raise ValueError("Training-energy measurement requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    return device


def infer_nvml_handle(pynvml, device: str, explicit_index: Optional[int]):
    if explicit_index is not None:
        return (
            pynvml.nvmlDeviceGetHandleByIndex(explicit_index),
            f"physical GPU {explicit_index}",
        )

    logical_index = int(device.split(":", 1)[1])
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return (
            pynvml.nvmlDeviceGetHandleByIndex(logical_index),
            f"physical GPU {logical_index}",
        )

    identifiers = [value.strip() for value in visible_devices.split(",")]
    if logical_index >= len(identifiers):
        raise ValueError(
            f"{device} is outside CUDA_VISIBLE_DEVICES={visible_devices!r}."
        )
    identifier = identifiers[logical_index]
    if identifier.isdigit():
        physical_index = int(identifier)
        return (
            pynvml.nvmlDeviceGetHandleByIndex(physical_index),
            f"physical GPU {physical_index}",
        )
    if identifier.startswith("GPU-"):
        return (
            pynvml.nvmlDeviceGetHandleByUUID(identifier),
            f"GPU UUID {identifier}",
        )
    raise ValueError(
        "Cannot infer the physical NVML device from "
        f"CUDA_VISIBLE_DEVICES={visible_devices!r}; pass --nvml-gpu-index."
    )


def initialize_energy_meter(
    device: str, explicit_index: Optional[int]
) -> NvmlEnergyMeter:
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires nvidia-ml-py (the pynvml module)."
        ) from exc

    pynvml.nvmlInit()
    try:
        handle, label = infer_nvml_handle(pynvml, device, explicit_index)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception:
        pynvml.nvmlShutdown()
        raise
    return NvmlEnergyMeter(pynvml=pynvml, handle=handle, device_label=label)


def synchronize(device: str) -> None:
    torch.cuda.synchronize(torch.device(device))


def checkpoint_default(config: dict, argument, key: str, fallback=None):
    return argument if argument is not None else config.get(key, fallback)


def resolve_run_config(args: argparse.Namespace, config: dict) -> dict:
    sequence_length = int(checkpoint_default(config, args.sequence_length, "bptt"))
    microbatch_size = int(
        checkpoint_default(config, args.microbatch_size, "batch_size")
    )
    aggregate_k_gradients = int(
        checkpoint_default(
            config, args.aggregate_k_gradients, "aggregate_k_gradients", 1
        )
    )
    epochs = int(checkpoint_default(config, args.epochs, "epochs", 200))
    microsteps_per_epoch = int(
        checkpoint_default(config, args.microsteps_per_epoch, "num_steps", 4672)
    )
    learning_rate = float(checkpoint_default(config, args.learning_rate, "lr", 1e-4))
    weight_decay_value = checkpoint_default(
        config, args.weight_decay, "weight_decay", 0.0
    )
    weight_decay = float(weight_decay_value or 0.0)
    shortcut_ce_weight = float(
        checkpoint_default(
            config,
            args.shortcut_ce_weight,
            "elastic_shortcut_ce_weight",
            0.1,
        )
    )
    consistency_weight = float(
        checkpoint_default(
            config,
            args.consistency_weight,
            "elastic_consistency_weight",
            0.1,
        )
    )
    temperature = float(
        checkpoint_default(
            config,
            args.distillation_temperature,
            "elastic_distillation_temperature",
            2.0,
        )
    )

    if not 0 < args.eval_position < sequence_length:
        raise ValueError("--eval-position must lie strictly inside the sequence.")
    if microbatch_size <= 0 or aggregate_k_gradients <= 0:
        raise ValueError("Microbatch size and gradient accumulation must be positive.")
    if epochs <= 0 or microsteps_per_epoch <= 0:
        raise ValueError("Epochs and microsteps per epoch must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative.")
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive.")
    for count_name in ("warmup_microbatches", "measured_microbatches"):
        count = int(getattr(args, count_name))
        if count % aggregate_k_gradients:
            raise ValueError(
                f"--{count_name.replace('_', '-')} ({count}) must be divisible by "
                f"aggregate_k_gradients ({aggregate_k_gradients})."
            )

    return {
        "sequence_length": sequence_length,
        "microbatch_size": microbatch_size,
        "aggregate_k_gradients": aggregate_k_gradients,
        "epochs": epochs,
        "microsteps_per_epoch": microsteps_per_epoch,
        "total_microsteps": epochs * microsteps_per_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "shortcut_ce_weight": shortcut_ce_weight,
        "consistency_weight": consistency_weight,
        "distillation_temperature": temperature,
    }


def make_operating_points(args: argparse.Namespace) -> List[OperatingPoint]:
    points = [
        OperatingPoint(
            model=f"looped_3x{depth}",
            role="individual_fixed_model",
            fixed_depth=depth,
        )
        for depth in args.fixed_depths
    ]
    points.extend(
        OperatingPoint(
            model=f"elastic_looped_3x{args.elastic_max_depth}_shortcut_{depth}",
            role="elastic_shortcut_operating_point",
            shortcut_depth=depth,
        )
        for depth in args.shortcut_depths
    )
    return points


def make_loss_fn(
    point: OperatingPoint,
    model: torch.nn.Module,
    inputs,
    targets: torch.Tensor,
    eval_position: int,
    args: argparse.Namespace,
    run_config: dict,
) -> Callable[[], torch.Tensor]:
    if point.fixed_depth is not None:
        return lambda: fixed_loss(
            model, inputs, targets, eval_position, point.fixed_depth
        )
    return lambda: elastic_loss(
        model,
        inputs,
        targets,
        eval_position,
        args.elastic_max_depth,
        int(point.shortcut_depth),
        run_config["shortcut_ce_weight"],
        run_config["consistency_weight"],
        run_config["distillation_temperature"],
    )


def has_nonfinite_grad(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def run_training_microbatches(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[], torch.Tensor],
    count: int,
    aggregate_k_gradients: int,
    gradient_clip: float,
) -> Tuple[float, float, int]:
    if count % aggregate_k_gradients:
        raise ValueError("Microbatch count must be divisible by gradient accumulation.")

    optimizer.zero_grad(set_to_none=True)
    first_loss = math.nan
    last_loss = math.nan
    optimizer_steps = 0
    for microbatch in range(count):
        loss = loss_fn()
        detached_loss = float(loss.detach().item())
        if not math.isfinite(detached_loss):
            raise RuntimeError(f"Non-finite loss at measured microbatch {microbatch}.")
        if microbatch == 0:
            first_loss = detached_loss
        last_loss = detached_loss

        (loss / aggregate_k_gradients).backward()
        if has_nonfinite_grad(model):
            raise RuntimeError(
                f"Non-finite gradient at measured microbatch {microbatch}."
            )
        if microbatch % aggregate_k_gradients == aggregate_k_gradients - 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError(
                    f"Non-finite gradient norm at measured microbatch {microbatch}."
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    return first_loss, last_loss, optimizer_steps


def measure_operating_point(
    point: OperatingPoint,
    repeat: int,
    order: int,
    model: torch.nn.Module,
    initial_state: dict,
    inputs,
    targets: torch.Tensor,
    meter: NvmlEnergyMeter,
    args: argparse.Namespace,
    run_config: dict,
    common_metadata: dict,
) -> dict:
    model.load_state_dict(initial_state)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config["learning_rate"],
        weight_decay=run_config["weight_decay"],
    )
    loss_fn = make_loss_fn(
        point, model, inputs, targets, args.eval_position, args, run_config
    )

    run_training_microbatches(
        model,
        optimizer,
        loss_fn,
        args.warmup_microbatches,
        run_config["aggregate_k_gradients"],
        args.gradient_clip,
    )
    synchronize(args.device)
    energy_before_millijoules = meter.read_millijoules()
    started = time.perf_counter()
    first_loss, last_loss, optimizer_steps = run_training_microbatches(
        model,
        optimizer,
        loss_fn,
        args.measured_microbatches,
        run_config["aggregate_k_gradients"],
        args.gradient_clip,
    )
    synchronize(args.device)
    elapsed_seconds = time.perf_counter() - started
    energy_after_millijoules = meter.read_millijoules()

    energy_joules = (
        energy_after_millijoules - energy_before_millijoules
    ) / 1000.0
    if energy_joules <= 0:
        raise RuntimeError(
            "NVML returned a nonpositive energy delta; the counter may be unsupported."
        )
    joules_per_microbatch = energy_joules / args.measured_microbatches
    full_training_kwh = (
        joules_per_microbatch * run_config["total_microsteps"] / JOULES_PER_KWH
    )
    co2e_kg = carbon_kg(
        full_training_kwh, args.carbon_intensity_g_per_kwh, args.pue
    )

    return {
        "repeat": repeat,
        "measurement_order": order,
        "model": point.model,
        "comparison_role": point.role,
        "fixed_depth": point.fixed_depth if point.fixed_depth is not None else "",
        "elastic_max_depth": args.elastic_max_depth,
        "shortcut_depth": (
            point.shortcut_depth if point.shortcut_depth is not None else ""
        ),
        "warmup_microbatches": args.warmup_microbatches,
        "measured_microbatches": args.measured_microbatches,
        "measured_optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed_seconds,
        "gpu_energy_joules": energy_joules,
        "joules_per_microbatch": joules_per_microbatch,
        "average_gpu_power_watts": energy_joules / elapsed_seconds,
        "full_training_microsteps": run_config["total_microsteps"],
        "estimated_full_training_gpu_kwh": full_training_kwh,
        "estimated_full_training_operational_co2e_kg": optional_number(co2e_kg),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "sequence_length": run_config["sequence_length"],
        "microbatch_size": run_config["microbatch_size"],
        "eval_position": args.eval_position,
        "aggregate_k_gradients": run_config["aggregate_k_gradients"],
        "learning_rate": run_config["learning_rate"],
        "weight_decay": run_config["weight_decay"],
        "gradient_clip": args.gradient_clip,
        "carbon_intensity_g_per_kwh": optional_number(
            args.carbon_intensity_g_per_kwh
        ),
        "pue": args.pue,
        **common_metadata,
    }


def optional_number(value: Optional[float]):
    return "" if value is None else value


def carbon_kg(
    energy_kwh: float,
    carbon_intensity_g_per_kwh: Optional[float],
    pue: float,
) -> Optional[float]:
    if carbon_intensity_g_per_kwh is None:
        return None
    return energy_kwh * pue * carbon_intensity_g_per_kwh / 1000.0


def write_csv(rows: List[dict], output_path: str) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_repeat_scenarios(
    raw_rows: List[dict], fixed_depths: Sequence[int], shortcut_depths: Sequence[int]
) -> List[dict]:
    scenarios = []
    repeats = sorted({int(row["repeat"]) for row in raw_rows})
    for repeat in repeats:
        repeat_rows = [row for row in raw_rows if int(row["repeat"]) == repeat]
        fixed_values: Dict[int, float] = {}
        for depth in fixed_depths:
            matches = [
                row
                for row in repeat_rows
                if row["comparison_role"] == "individual_fixed_model"
                and int(row["fixed_depth"]) == depth
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one fixed-depth {depth} row for repeat {repeat}."
                )
            value = float(matches[0]["joules_per_microbatch"])
            fixed_values[depth] = value
            scenarios.append(
                {
                    "repeat": repeat,
                    "scenario": f"looped_3x{depth}",
                    "comparison_role": "individual_fixed_model",
                    "joules_per_microbatch_equivalent": value,
                }
            )

        elastic_matches = [
            row
            for row in repeat_rows
            if row["comparison_role"] == "elastic_shortcut_operating_point"
        ]
        measured_shortcuts = sorted(int(row["shortcut_depth"]) for row in elastic_matches)
        if measured_shortcuts != sorted(shortcut_depths):
            raise ValueError(
                f"Elastic shortcut rows for repeat {repeat} are incomplete or duplicated."
            )
        elastic_value = statistics.fmean(
            float(row["joules_per_microbatch"]) for row in elastic_matches
        )
        fixed_total = sum(fixed_values.values())
        scenarios.extend(
            [
                {
                    "repeat": repeat,
                    "scenario": "fixed_models_total",
                    "comparison_role": "sum_of_fixed_models",
                    "joules_per_microbatch_equivalent": fixed_total,
                },
                {
                    "repeat": repeat,
                    "scenario": "elastic_looped_3x6_expected",
                    "comparison_role": "elastic_model",
                    "joules_per_microbatch_equivalent": elastic_value,
                },
                {
                    "repeat": repeat,
                    "scenario": "elastic_reduction_vs_fixed_suite_percent",
                    "comparison_role": "comparison",
                    "joules_per_microbatch_equivalent": 100.0
                    * (1.0 - elastic_value / fixed_total),
                },
            ]
        )
    return scenarios


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def mean_sd_ci95(values: Sequence[float]) -> Tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    standard_deviation = statistics.stdev(values)
    degrees_freedom = len(values) - 1
    critical = T_CRITICAL_975.get(degrees_freedom, 1.96)
    half_width = critical * standard_deviation / math.sqrt(len(values))
    return mean, standard_deviation, mean - half_width, mean + half_width


def build_summary_rows(
    raw_rows: List[dict],
    fixed_depths: Sequence[int],
    shortcut_depths: Sequence[int],
    total_microsteps: int,
    carbon_intensity_g_per_kwh: Optional[float],
    pue: float,
) -> List[dict]:
    scenarios = build_repeat_scenarios(raw_rows, fixed_depths, shortcut_depths)
    scenario_order = [f"looped_3x{depth}" for depth in fixed_depths] + [
        "fixed_models_total",
        "elastic_looped_3x6_expected",
        "elastic_reduction_vs_fixed_suite_percent",
    ]
    rows = []
    for scenario_name in scenario_order:
        matching = [row for row in scenarios if row["scenario"] == scenario_name]
        values = [float(row["joules_per_microbatch_equivalent"]) for row in matching]
        mean, standard_deviation, ci_low, ci_high = mean_sd_ci95(values)
        is_comparison = matching[0]["comparison_role"] == "comparison"
        energy_kwh = None if is_comparison else mean * total_microsteps / JOULES_PER_KWH
        rows.append(
            {
                "scenario": scenario_name,
                "comparison_role": matching[0]["comparison_role"],
                "repetitions": len(values),
                "mean_joules_per_microbatch_equivalent": (
                    "" if is_comparison else mean
                ),
                "sd_joules_per_microbatch_equivalent": (
                    "" if is_comparison else standard_deviation
                ),
                "ci95_low_joules_per_microbatch_equivalent": (
                    "" if is_comparison else ci_low
                ),
                "ci95_high_joules_per_microbatch_equivalent": (
                    "" if is_comparison else ci_high
                ),
                "full_training_microsteps": "" if is_comparison else total_microsteps,
                "estimated_full_training_gpu_kwh": optional_number(energy_kwh),
                "estimated_full_training_operational_co2e_kg": optional_number(
                    None
                    if energy_kwh is None
                    else carbon_kg(energy_kwh, carbon_intensity_g_per_kwh, pue)
                ),
                "mean_elastic_reduction_vs_fixed_suite_percent": (
                    mean if is_comparison else ""
                ),
                "sd_elastic_reduction_vs_fixed_suite_percent": (
                    standard_deviation if is_comparison else ""
                ),
                "ci95_low_elastic_reduction_percent": ci_low if is_comparison else "",
                "ci95_high_elastic_reduction_percent": ci_high if is_comparison else "",
                "carbon_intensity_g_per_kwh": optional_number(
                    carbon_intensity_g_per_kwh
                ),
                "pue": pue,
            }
        )
    return rows


def print_configuration(args: argparse.Namespace, run_config: dict, gpu: dict) -> None:
    print("Training-energy benchmark configuration", flush=True)
    print(f"  checkpoint: {args.checkpoint}", flush=True)
    print(
        f"  GPU: {gpu['gpu_name']} ({gpu['nvml_device']}, {gpu['gpu_uuid']})",
        flush=True,
    )
    print(
        f"  tensor shape: sequence={run_config['sequence_length']}, "
        f"microbatch={run_config['microbatch_size']}, eval_position={args.eval_position}",
        flush=True,
    )
    print(
        f"  accumulation={run_config['aggregate_k_gradients']}, "
        f"warmup={args.warmup_microbatches}, measured={args.measured_microbatches}, "
        f"repeats={args.repeats}",
        flush=True,
    )
    print(
        f"  extrapolation: {run_config['epochs']} epochs x "
        f"{run_config['microsteps_per_epoch']} microbatches = "
        f"{run_config['total_microsteps']} microbatches",
        flush=True,
    )


def print_final_summary(summary_rows: List[dict], args: argparse.Namespace) -> None:
    print("\nMeasured training-energy summary", flush=True)
    for row in summary_rows:
        if row["comparison_role"] == "comparison":
            print(
                "  Elastic versus all fixed models, training-GPU energy reduction: "
                f"{row['mean_elastic_reduction_vs_fixed_suite_percent']:.2f}% "
                f"(95% CI {row['ci95_low_elastic_reduction_percent']:.2f}% to "
                f"{row['ci95_high_elastic_reduction_percent']:.2f}%).",
                flush=True,
            )
        else:
            print(
                f"  {row['scenario']:<30} "
                f"{row['estimated_full_training_gpu_kwh']:.6f} kWh",
                flush=True,
            )
    if args.carbon_intensity_g_per_kwh is None:
        print(
            "  Absolute CO2e omitted: pass --carbon-intensity-g-per-kwh to compute it.",
            flush=True,
        )
    print(f"  raw CSV: {args.raw_output_csv}", flush=True)
    print(f"  summary CSV: {args.summary_output_csv}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.device = normalize_device(args.device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")


    meter = initialize_energy_meter(args.device, args.nvml_gpu_index)
    try:
        # Check before PyTorch creates this process's CUDA context. NVML may
        # report host-namespace PIDs that cannot be compared with os.getpid()
        # from inside a container.
        other_processes = meter.compute_process_ids()
        if other_processes and not args.allow_shared_gpu:
            raise RuntimeError(
                f"{meter.device_label} has other compute processes {other_processes}. "
                "Use a dedicated GPU or pass --allow-shared-gpu and accept biased results."
            )

        torch.cuda.set_device(torch.device(args.device))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        loaded, checkpoint_config = load_model_only_inference(
            ".", str(checkpoint), args.device, model_name="looped_transformer"
        )
        model = loaded[2]
        model.train()
        run_config = resolve_run_config(args, checkpoint_config)
        inputs, targets = make_synthetic_batch(
            checkpoint_config,
            args.device,
            run_config["sequence_length"],
            run_config["microbatch_size"],
            args.eval_position,
        )
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        gpu_metadata = meter.metadata()
        common_metadata = {
            "checkpoint": str(checkpoint),
            **gpu_metadata,
        }
        print_configuration(args, run_config, gpu_metadata)
        if other_processes:
            print(
                f"WARNING: shared GPU with compute processes {other_processes}; "
                "energy results include their consumption.",
                flush=True,
            )

        raw_rows: List[dict] = []
        base_points = make_operating_points(args)
        for repeat in range(1, args.repeats + 1):
            points = list(base_points)
            if not args.fixed_order:
                random.Random(args.seed + repeat).shuffle(points)
            for order, point in enumerate(points, start=1):
                print(
                    f"repeat {repeat}/{args.repeats}, point {order}/{len(points)}: "
                    f"{point.model}",
                    flush=True,
                )
                row = measure_operating_point(
                    point,
                    repeat,
                    order,
                    model,
                    initial_state,
                    inputs,
                    targets,
                    meter,
                    args,
                    run_config,
                    common_metadata,
                )
                raw_rows.append(row)
                write_csv(raw_rows, args.raw_output_csv)
                print(
                    f"  {row['joules_per_microbatch']:.3f} J/microbatch, "
                    f"{row['average_gpu_power_watts']:.1f} W average",
                    flush=True,
                )
    finally:
        meter.shutdown()

    summary_rows = build_summary_rows(
        raw_rows,
        args.fixed_depths,
        args.shortcut_depths,
        run_config["total_microsteps"],
        args.carbon_intensity_g_per_kwh,
        args.pue,
    )
    write_csv(summary_rows, args.summary_output_csv)
    print_final_summary(summary_rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
