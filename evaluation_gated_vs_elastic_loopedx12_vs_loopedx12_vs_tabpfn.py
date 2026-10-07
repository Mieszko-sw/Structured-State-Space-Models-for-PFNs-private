"""Compare four TabPFN variants at 12 effective Transformer layers.

The comparison includes:

* gated recurrent Transformer: one prelude layer, one shared core repeated ten
  times, and one coda layer (three physical blocks, 12 effective layers);
* elastic looped x12: one physical TabPFN block repeated 12 times;
* standard looped x12: one physical TabPFN block repeated 12 times;
* original TabPFN: 12 independent Transformer layers.

By default, the models are evaluated on splits 1-5 of the project's 30-dataset
OpenML-CC18 subset. AUC-ROC is averaged across splits within each dataset and
then macro-averaged across datasets, giving every dataset equal weight.

Outputs:

* a long CSV with one row per model, dataset, and split;
* a wide per-dataset CSV with split scores, means, standard deviations, and
  pairwise mean-score differences;
* a summary CSV with macro-average scores, normal-approximation 95% confidence
  intervals across datasets, and paired differences/win counts.
"""

import argparse
import itertools
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


EFFECTIVE_LAYERS = 12

GATED_MODEL = "gated_recurrent_transformer"
ELASTIC_MODEL = "elastic_looped_x12"
LOOPED_MODEL = "looped_x12"
TABPFN_MODEL = "tabpfn"
MODEL_ORDER = (GATED_MODEL, ELASTIC_MODEL, LOOPED_MODEL, TABPFN_MODEL)

DEFAULT_GATED_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_gated_recurrent_transformer_1pre_1core10x_1coda_latest.cpkt"
)
DEFAULT_ELASTIC_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_1physical_core12x_latest.cpkt"
)
DEFAULT_LOOPED_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_new_looped_transformer_1physical_core12x_latest.cpkt"
)
DEFAULT_TABPFN_CHECKPOINT = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"

# ``None`` leaves the ordinary 12-layer TabPFN at its native depth.
INFERENCE_LOOPS = {
    GATED_MODEL: 10,
    ELASTIC_MODEL: 12,
    LOOPED_MODEL: 12,
    TABPFN_MODEL: None,
}
PHYSICAL_LAYERS = {
    GATED_MODEL: 3,
    ELASTIC_MODEL: 1,
    LOOPED_MODEL: 1,
    TABPFN_MODEL: 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs (default: the project's 30-dataset list).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        choices=benchmark.SPLIT_NUMBERS,
        default=benchmark.SPLIT_NUMBERS,
        help="Splits averaged within each dataset (default: 1 2 3 4 5).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--gated-checkpoint", default=DEFAULT_GATED_CHECKPOINT)
    parser.add_argument(
        "--elastic-checkpoint", default=DEFAULT_ELASTIC_CHECKPOINT
    )
    parser.add_argument("--looped-checkpoint", default=DEFAULT_LOOPED_CHECKPOINT)
    parser.add_argument("--tabpfn-checkpoint", default=DEFAULT_TABPFN_CHECKPOINT)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "gated_vs_elastic_loopedx12_vs_loopedx12_vs_tabpfn_raw.csv",
        ),
    )
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "gated_vs_elastic_loopedx12_vs_loopedx12_vs_tabpfn_per_dataset.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "gated_vs_elastic_loopedx12_vs_loopedx12_vs_tabpfn_summary.csv",
        ),
    )
    return parser.parse_args()


def build_model_specs(args):
    return {
        GATED_MODEL: {
            "path": args.gated_checkpoint,
            # The custom loader selects the gated class from the checkpoint flag.
            "loader_type": "looped_transformer",
            "method_name": "transformer",
        },
        ELASTIC_MODEL: {
            "path": args.elastic_checkpoint,
            "loader_type": "looped_transformer",
            "method_name": "transformer",
        },
        LOOPED_MODEL: {
            "path": args.looped_checkpoint,
            "loader_type": "looped_transformer",
            "method_name": "transformer",
        },
        TABPFN_MODEL: {
            "path": args.tabpfn_checkpoint,
            "loader_type": "tabpfn",
            "method_name": "transformer",
        },
    }


def ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def sample_std(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def mean_ci95(values):
    """Return mean and a normal-approximation 95% CI across datasets."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / np.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def validate_config(model_name, config):
    """Fail early if a supplied checkpoint is not the advertised architecture."""
    if int(config["nlayers"]) != EFFECTIVE_LAYERS:
        raise ValueError(
            f"{model_name}: expected nlayers={EFFECTIVE_LAYERS}, "
            f"got {config['nlayers']}"
        )

    if model_name == GATED_MODEL:
        fields = {
            "gated_recurrent_transformer": True,
            "gated_warmup_layers": 1,
            "gated_core_layers": 1,
            "gated_exit_layers": 1,
            "gated_max_loops": 10,
        }
        for key, expected in fields.items():
            actual = config.get(key)
            if actual != expected:
                raise ValueError(
                    f"{model_name}: expected {key}={expected!r}, got {actual!r}"
                )
    elif model_name in (ELASTIC_MODEL, LOOPED_MODEL):
        fields = {
            "looped_warmup_layers": 0,
            "looped_core_layers": 1,
            "looped_exit_layers": 0,
        }
        for key, expected in fields.items():
            actual = config.get(key)
            if actual != expected:
                raise ValueError(
                    f"{model_name}: expected {key}={expected!r}, got {actual!r}"
                )
        repeat_pattern = config.get("looped_core_repeat_pattern")
        if repeat_pattern not in ([12], (12,)):
            raise ValueError(
                f"{model_name}: expected looped_core_repeat_pattern=[12], "
                f"got {repeat_pattern!r}"
            )


def evaluate_one(model, config, model_name, specs, dataset_list, args, split):
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(
            f"{model_name}: expected one evaluation position, got {eval_positions}"
        )

    num_loops = INFERENCE_LOOPS[model_name]
    depth_argument = {} if num_loops is None else {"num_loops": num_loops}
    with torch.no_grad():
        result = benchmark.evaluate(
            datasets=dataset_list,
            bptt=benchmark.BPTT,
            eval_positions=eval_positions,
            metric_used=benchmark.METRIC_USED,
            model=model,
            device=args.device,
            method_name=specs[model_name]["method_name"],
            max_time=args.max_time,
            split_number=split,
            jrt_prompt=False,
            random_premutation=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            **depth_argument,
        )
    return benchmark.extract_metric(result)


def evaluate_all(models, specs, datasets, args):
    rows = []
    for did, dataset_list in datasets.items():
        dataset_name, x, _, _, _, _ = dataset_list[0]
        for split in args.splits:
            for model_name in MODEL_ORDER:
                model, config = models[model_name]
                score = evaluate_one(
                    model,
                    config,
                    model_name,
                    specs,
                    dataset_list,
                    args,
                    split,
                )
                row = {
                    "did": did,
                    "dataset_name": dataset_name,
                    "num_samples": int(x.shape[0]),
                    "num_features": int(x.shape[1]),
                    "split_number": split,
                    "model": model_name,
                    "physical_transformer_layers": PHYSICAL_LAYERS[model_name],
                    "num_recurrent_loops": INFERENCE_LOOPS[model_name],
                    "effective_transformer_layers": EFFECTIVE_LAYERS,
                    "auc_roc": score,
                }
                rows.append(row)
                print(
                    f"model={model_name} did={did} split={split} "
                    f"auc_roc={score:.6f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def make_per_dataset(raw):
    metadata_columns = ["did", "dataset_name", "num_samples", "num_features"]
    metadata = raw[metadata_columns].drop_duplicates("did").set_index("did")
    per_dataset = metadata.copy()

    for model_name in MODEL_ORDER:
        model_rows = raw[raw["model"] == model_name]
        grouped = model_rows.groupby("did")["auc_roc"]
        per_dataset[f"{model_name}_mean_auc_roc"] = grouped.mean()
        per_dataset[f"{model_name}_std_auc_roc"] = grouped.std(ddof=1)
        split_scores = model_rows.pivot(
            index="did", columns="split_number", values="auc_roc"
        )
        for split in sorted(model_rows["split_number"].unique()):
            per_dataset[f"split_{split}_{model_name}_auc_roc"] = split_scores[
                split
            ]

    for first, second in itertools.combinations(MODEL_ORDER, 2):
        per_dataset[f"{first}_minus_{second}_mean_auc_roc"] = (
            per_dataset[f"{first}_mean_auc_roc"]
            - per_dataset[f"{second}_mean_auc_roc"]
        )

    per_dataset.insert(3, "split_count", raw["split_number"].nunique())
    return per_dataset.reset_index().sort_values("did")


def make_summary(per_dataset):
    row = {
        "dataset_count": int(per_dataset["did"].nunique()),
        "splits_per_dataset": int(per_dataset["split_count"].iloc[0]),
        "effective_transformer_layers": EFFECTIVE_LAYERS,
    }

    for model_name in MODEL_ORDER:
        values = per_dataset[f"{model_name}_mean_auc_roc"]
        mean, std, ci_low, ci_high = mean_ci95(values)
        row[f"{model_name}_average_auc_roc"] = mean
        row[f"{model_name}_std_auc_roc_across_datasets"] = std
        row[f"{model_name}_ci95_low"] = ci_low
        row[f"{model_name}_ci95_high"] = ci_high

    for first, second in itertools.combinations(MODEL_ORDER, 2):
        prefix = f"{first}_minus_{second}"
        differences = per_dataset[f"{prefix}_mean_auc_roc"]
        mean, std, ci_low, ci_high = mean_ci95(differences)
        row[f"{prefix}_average_auc_roc"] = mean
        row[f"{prefix}_std_auc_roc_across_datasets"] = std
        row[f"{prefix}_ci95_low"] = ci_low
        row[f"{prefix}_ci95_high"] = ci_high
        row[f"{first}_wins_vs_{second}"] = int((differences > 0).sum())
        row[f"{first}_ties_vs_{second}"] = int((differences == 0).sum())
        row[f"{first}_losses_vs_{second}"] = int((differences < 0).sum())

    return pd.DataFrame([row])


def main():
    args = parse_args()
    args.splits = list(dict.fromkeys(args.splits))
    args.device = benchmark.normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    specs = build_model_specs(args)
    for model_name in MODEL_ORDER:
        checkpoint = specs[model_name]["path"]
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint for {model_name} does not exist: {checkpoint}"
            )

    eval_helper = EvalHelper()
    dids = (
        args.dids
        if args.dids is not None
        else eval_helper.openml_cc18_dids_small
    )
    datasets = benchmark.prepare_datasets(dids)
    if len(datasets) != len(dids):
        raise RuntimeError(
            f"Requested {len(dids)} datasets but only {len(datasets)} "
            "passed preparation."
        )

    # ``benchmark.load_model`` reads its model registry from that module.
    benchmark.MODELS = specs
    models = {
        model_name: benchmark.load_model(model_name, args.device)
        for model_name in MODEL_ORDER
    }
    for model_name, (_, config) in models.items():
        validate_config(model_name, config)

    raw = evaluate_all(models, specs, datasets, args)
    per_dataset = make_per_dataset(raw)
    summary = make_summary(per_dataset)

    for path in (args.raw_csv, args.per_dataset_csv, args.summary_csv):
        ensure_parent_directory(path)
    raw.to_csv(args.raw_csv, index=False)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print("\nMacro-average AUC-ROC across OpenML datasets:")
    columns = [f"{name}_average_auc_roc" for name in MODEL_ORDER]
    print(summary[columns].to_string(index=False))
    print(f"\nSaved split-level scores to {args.raw_csv}")
    print(f"Saved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved macro summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
