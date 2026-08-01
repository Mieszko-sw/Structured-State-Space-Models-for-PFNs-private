import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


MODEL_PATHS = {
    "hybrid_8l": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
    "THx4": (
        "tabpfn/models_diff/"
        "callback_tabpfn_hydra_x4_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
    ),
    "HHTTTTHH": (
        "tabpfn/models_diff/"
        "callback_hydra2_tabpfn4_hydra2_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
    ),
    "HTTTTTTH": (
        "tabpfn/models_diff/"
        "callback_hydra1_tabpfn6_hydra1_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
    ),
    "TTHHHHTT": (
        "tabpfn/models_diff/"
        "callback_tabpfn2_hydra4_tabpfn2_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
    ),
    "THHHHHHT": (
        "tabpfn/models_diff/"
        "callback_tabpfn1_hydra6_tabpfn1_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
    ),
    "hydra_16M": "tabpfn/models_diff/callback_pure_hydra_9_layers_latest.cpkt",
    "hydra_small": "tabpfn/models_diff/hydra_small.cpkt",
    "tabpfn": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
}

MODEL_TYPES = {
    "hybrid_8l": "hybrid",
    "THx4": "hybrid",
    "HHTTTTHH": "hybrid",
    "HTTTTTTH": "hybrid",
    "TTHHHHTT": "hybrid",
    "THHHHHHT": "hybrid",
    "hydra_16M": "hydra",
}

PREDICTION_METHODS = {
    "hybrid_8l": "transformer",
    "THx4": "transformer",
    "HHTTTTHH": "transformer",
    "HTTTTTTH": "transformer",
    "TTHHHHTT": "transformer",
    "THHHHHHT": "transformer",
    "hydra_16M": "hydra",
    "hydra_small": "hydra",
    "tabpfn": "transformer",
}

METRIC_USED = tabular_metrics.auc_metric
RAW_COLUMNS = [
    "model",
    "table_size",
    "num_features",
    "eval_position",
    "run",
    "status",
    "error",
    "synchronized_elapsed_seconds",
    "internal_unsynchronized_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized inference speed for all trained mixed Hydra/TabPFN "
            "8-layer layouts, Hydra 9L (16M), Hydra Small, and TabPFN."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_PATHS),
        choices=list(MODEL_PATHS),
    )
    parser.add_argument(
        "--table-sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join("result_csvs", "evaluation_speed_raw.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join("result_csvs", "evaluation_speed_summary.csv"),
    )
    parser.add_argument(
        "--internal-time-csv",
        default=os.path.join("result_csvs", "evaluation_speed_internal_time_summary.csv"),
    )
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    return device


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def clear_device_cache(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom_error(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def format_error(error):
    return str(error).splitlines()[0]


def print_parameter_count(model_name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} parameters: {total_params:,} total ({trainable_params:,} trainable)")


def make_dummy_dataset(table_size, num_features, num_classes, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(table_size, num_features, generator=generator)
    y = torch.arange(table_size) % num_classes
    permutation = torch.randperm(table_size, generator=generator)
    x = x[permutation]
    y = y[permutation]
    return [f"dummy_{table_size}x{num_features}", x, y, [], None, None]


def load_model(model_name, device):
    if model_name == "hydra_small":
        loaded, config, _ = hydra_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=MODEL_PATHS[model_name],
        )
    elif model_name in MODEL_TYPES:
        loaded, config = load_model_only_inference(
            ".",
            MODEL_PATHS[model_name],
            device,
            model_name=MODEL_TYPES[model_name],
        )
    elif model_name == "tabpfn":
        loaded, config, _ = transformer_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=MODEL_PATHS[model_name],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = loaded[2]
    model.eval()
    print_parameter_count(model_name, model)
    return model, config


def extract_internal_inference_time(result, dataset_name, eval_position):
    key = f"{dataset_name}_time_at_{eval_position}"
    return result.get(key)


def run_single_measurement(model, model_name, dataset, table_size, device):
    dataset_name = dataset[0]
    eval_position = table_size // 2

    synchronize(device)
    start = time.perf_counter()
    result = evaluate(
        datasets=[dataset],
        bptt=table_size,
        eval_positions=[eval_position],
        metric_used=METRIC_USED,
        model=model,
        device=device,
        method_name=PREDICTION_METHODS[model_name],
        jrt_prompt=False,
        random_premutation=False,
        single_evaluation_prompt=False,
        permutation_bagging=1,
        sample_bagging=0,
    )
    synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    return elapsed_seconds, extract_internal_inference_time(result, dataset_name, eval_position)


def run_warmups(models, datasets, args):
    print("Running warmup passes...", flush=True)
    skipped_pairs = set()
    for table_size, dataset in datasets.items():
        for model_name, model in models.items():
            for warmup_idx in range(args.warmup_runs):
                try:
                    run_single_measurement(model, model_name, dataset, table_size, args.device)
                except RuntimeError as error:
                    if not is_oom_error(error):
                        raise
                    clear_device_cache(args.device)
                    skipped_pairs.add((model_name, table_size))
                    print(
                        f"skip warmup model={model_name} table_size={table_size} "
                        f"run={warmup_idx + 1} error={format_error(error)}",
                        flush=True,
                    )
                    break
                print(
                    f"warmup model={model_name} table_size={table_size} run={warmup_idx + 1}",
                    flush=True,
                )
    return skipped_pairs


def make_failure_row(model_name, table_size, args, run, status, error):
    return {
        "model": model_name,
        "table_size": table_size,
        "num_features": args.num_features,
        "eval_position": table_size // 2,
        "run": run,
        "status": status,
        "error": error,
        "synchronized_elapsed_seconds": np.nan,
        "internal_unsynchronized_seconds": np.nan,
    }


def run_timed_measurements(models, datasets, args, skipped_pairs):
    rng = random.Random(args.seed)
    rows = [
        make_failure_row(
            model_name,
            table_size,
            args,
            run=0,
            status="skipped_warmup_oom",
            error="warmup CUDA out of memory",
        )
        for model_name, table_size in sorted(skipped_pairs)
    ]
    failed_pairs = set(skipped_pairs)

    for table_size, dataset in datasets.items():
        for run_idx in range(args.timed_runs):
            model_names = list(models)
            rng.shuffle(model_names)

            for model_name in model_names:
                if (model_name, table_size) in failed_pairs:
                    continue
                try:
                    elapsed_seconds, internal_seconds = run_single_measurement(
                        models[model_name],
                        model_name,
                        dataset,
                        table_size,
                        args.device,
                    )
                except RuntimeError as error:
                    if not is_oom_error(error):
                        raise
                    clear_device_cache(args.device)
                    failed_pairs.add((model_name, table_size))
                    rows.append(
                        make_failure_row(
                            model_name,
                            table_size,
                            args,
                            run=run_idx + 1,
                            status="timed_oom",
                            error=format_error(error),
                        )
                    )
                    print(
                        f"skip timed model={model_name} table_size={table_size} "
                        f"run={run_idx + 1} error={format_error(error)}",
                        flush=True,
                    )
                    continue
                row = {
                    "model": model_name,
                    "table_size": table_size,
                    "num_features": args.num_features,
                    "eval_position": table_size // 2,
                    "run": run_idx + 1,
                    "status": "ok",
                    "error": "",
                    "synchronized_elapsed_seconds": elapsed_seconds,
                    "internal_unsynchronized_seconds": internal_seconds,
                }
                rows.append(row)
                print(
                    f"timed model={model_name} table_size={table_size} "
                    f"run={run_idx + 1} seconds={elapsed_seconds:.6f}",
                    flush=True,
                )

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw_df, value_column):
    raw_df = raw_df[raw_df["status"] == "ok"]
    return (
        raw_df.groupby(["model", "table_size", "num_features", "eval_position"])[value_column]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )


def write_csv(df, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    datasets = {
        table_size: make_dummy_dataset(
            table_size,
            args.num_features,
            args.num_classes,
            args.seed + table_size,
        )
        for table_size in args.table_sizes
    }

    models = {model_name: load_model(model_name, args.device)[0] for model_name in args.models}

    skipped_pairs = run_warmups(models, datasets, args)
    raw_df = run_timed_measurements(models, datasets, args, skipped_pairs)
    summary_df = summarize(raw_df, "synchronized_elapsed_seconds")
    internal_summary_df = summarize(raw_df, "internal_unsynchronized_seconds")

    write_csv(raw_df, args.raw_csv)
    write_csv(summary_df, args.summary_csv)
    write_csv(internal_summary_df, args.internal_time_csv)

    print("\nSynchronized elapsed seconds summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved synchronized summary to {args.summary_csv}")
    print(f"Saved internal unsynchronized summary to {args.internal_time_csv}")


if __name__ == "__main__":
    main()
