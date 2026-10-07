"""Compare LoopFormer, two looped x12 models, and TabPFN across depths 1-12.

LoopFormer, elastic looped x12, and standard looped x12 each have a single
physical TabPFN Transformer block and are evaluated with the same requested
number of recurrent applications. Original TabPFN has 12 independent layers
and no recurrent-depth control, so it is evaluated once per dataset/split and
reported as the same fixed 12-layer reference at every comparison depth.

The default benchmark uses splits 1-5 of the project's 30-dataset OpenML-CC18
subset. AUC-ROC is averaged across splits within each dataset and then
macro-averaged across datasets, so every dataset receives equal weight.

Outputs:

* a long CSV with one row per depth, dataset, split, and model;
* a wide per-dataset/depth CSV with split scores, means, standard deviations,
  and every pairwise mean-score difference;
* a depth-summary CSV with macro averages, 95% confidence intervals across
  datasets, paired differences, and dataset win/tie/loss counts.
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


MAX_DEPTH = 12

LOOPFORMER_MODEL = "loopformer"
ELASTIC_MODEL = "elastic_looped_x12"
LOOPED_MODEL = "looped_x12"
TABPFN_MODEL = "tabpfn"
RECURRENT_MODELS = (LOOPFORMER_MODEL, ELASTIC_MODEL, LOOPED_MODEL)
MODEL_ORDER = RECURRENT_MODELS + (TABPFN_MODEL,)

DEFAULT_LOOPFORMER_CHECKPOINT = (
    "tabpfn/models_diff/callback_loopformer_1physical_core12x_latest.cpkt"
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

PHYSICAL_TRANSFORMER_LAYERS = {
    LOOPFORMER_MODEL: 1,
    ELASTIC_MODEL: 1,
    LOOPED_MODEL: 1,
    TABPFN_MODEL: 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        choices=range(1, MAX_DEPTH + 1),
        default=list(range(1, MAX_DEPTH + 1)),
        metavar="N",
        help="Recurrent depths to compare (default: 1 through 12).",
    )
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
    parser.add_argument(
        "--loopformer-checkpoint", default=DEFAULT_LOOPFORMER_CHECKPOINT
    )
    parser.add_argument(
        "--elastic-checkpoint", default=DEFAULT_ELASTIC_CHECKPOINT
    )
    parser.add_argument("--looped-checkpoint", default=DEFAULT_LOOPED_CHECKPOINT)
    parser.add_argument("--tabpfn-checkpoint", default=DEFAULT_TABPFN_CHECKPOINT)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "loopformer_vs_elastic_loopedx12_vs_loopedx12_vs_tabpfn_depths_raw.csv",
        ),
    )
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "loopformer_vs_elastic_loopedx12_vs_loopedx12_vs_tabpfn_depths_per_dataset.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "loopformer_vs_elastic_loopedx12_vs_loopedx12_vs_tabpfn_depths_summary.csv",
        ),
    )
    return parser.parse_args()


def build_model_specs(args):
    return {
        LOOPFORMER_MODEL: {
            "path": args.loopformer_checkpoint,
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
    """Return mean, sample standard deviation, and a normal 95% CI."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / np.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def validate_config(model_name, config):
    """Reject checkpoints whose architecture does not match this comparison."""
    if int(config["nlayers"]) != MAX_DEPTH:
        raise ValueError(
            f"{model_name}: expected nlayers={MAX_DEPTH}, got {config['nlayers']}"
        )

    if model_name == LOOPFORMER_MODEL:
        expected = {
            "loopformer_shortcut_training": True,
            "loopformer_core_layers": 1,
            "loopformer_max_loops": MAX_DEPTH,
        }
    elif model_name in (ELASTIC_MODEL, LOOPED_MODEL):
        expected = {
            "looped_warmup_layers": 0,
            "looped_core_layers": 1,
            "looped_exit_layers": 0,
        }
    else:
        expected = {}

    for key, expected_value in expected.items():
        actual = config.get(key)
        if actual != expected_value:
            raise ValueError(
                f"{model_name}: expected {key}={expected_value!r}, got {actual!r}"
            )

    if model_name in (ELASTIC_MODEL, LOOPED_MODEL):
        repeat_pattern = config.get("looped_core_repeat_pattern")
        if repeat_pattern not in ([MAX_DEPTH], (MAX_DEPTH,)):
            raise ValueError(
                f"{model_name}: expected looped_core_repeat_pattern="
                f"[{MAX_DEPTH}], got {repeat_pattern!r}"
            )
    if model_name == ELASTIC_MODEL and not config.get(
        "elastic_shortcut_training", False
    ):
        raise ValueError(
            f"{model_name}: checkpoint was not trained with elastic shortcuts"
        )


def evaluate_one(
    model,
    config,
    model_name,
    specs,
    dataset_list,
    args,
    split,
    depth=None,
):
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(
            f"{model_name}: expected one evaluation position, got {eval_positions}"
        )

    depth_argument = {} if depth is None else {"num_loops": depth}
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


def make_raw_row(
    did,
    dataset_name,
    x,
    depth,
    split,
    model_name,
    score,
):
    is_tabpfn = model_name == TABPFN_MODEL
    return {
        "comparison_depth": depth,
        "did": did,
        "dataset_name": dataset_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "split_number": split,
        "model": model_name,
        "physical_transformer_layers": PHYSICAL_TRANSFORMER_LAYERS[model_name],
        "num_recurrent_loops": np.nan if is_tabpfn else depth,
        "model_effective_transformer_layers": MAX_DEPTH if is_tabpfn else depth,
        "is_fixed_depth_reference": is_tabpfn,
        "auc_roc": score,
    }


def evaluate_all(models, specs, datasets, args):
    rows = []
    for did, dataset_list in datasets.items():
        dataset_name, x, _, _, _, _ = dataset_list[0]

        # TabPFN is independent of comparison depth, so compute it only once.
        tabpfn_scores = {}
        tabpfn, tabpfn_config = models[TABPFN_MODEL]
        for split in args.splits:
            score = evaluate_one(
                tabpfn,
                tabpfn_config,
                TABPFN_MODEL,
                specs,
                dataset_list,
                args,
                split,
            )
            tabpfn_scores[split] = score
            print(
                f"model={TABPFN_MODEL} did={did} split={split} "
                f"auc_roc={score:.6f} fixed_depth_reference=true",
                flush=True,
            )

        for depth in args.depths:
            for split in args.splits:
                for model_name in RECURRENT_MODELS:
                    model, config = models[model_name]
                    score = evaluate_one(
                        model,
                        config,
                        model_name,
                        specs,
                        dataset_list,
                        args,
                        split,
                        depth=depth,
                    )
                    rows.append(
                        make_raw_row(
                            did,
                            dataset_name,
                            x,
                            depth,
                            split,
                            model_name,
                            score,
                        )
                    )
                    print(
                        f"model={model_name} depth={depth} did={did} "
                        f"split={split} auc_roc={score:.6f}",
                        flush=True,
                    )

                rows.append(
                    make_raw_row(
                        did,
                        dataset_name,
                        x,
                        depth,
                        split,
                        TABPFN_MODEL,
                        tabpfn_scores[split],
                    )
                )

        # Preserve all fully completed datasets if a long sweep is interrupted.
        pd.DataFrame(rows).to_csv(args.raw_csv, index=False)

    return pd.DataFrame(rows)


def make_per_dataset(raw):
    rows = []
    for (depth, did), group in raw.groupby(
        ["comparison_depth", "did"], sort=True
    ):
        first = group.iloc[0]
        row = {
            "comparison_depth": int(depth),
            "did": int(did),
            "dataset_name": first["dataset_name"],
            "num_samples": int(first["num_samples"]),
            "num_features": int(first["num_features"]),
            "split_count": int(group["split_number"].nunique()),
        }
        means = {}
        for model_name in MODEL_ORDER:
            model_rows = group[group["model"] == model_name].sort_values(
                "split_number"
            )
            scores = model_rows["auc_roc"].to_numpy(dtype=float)
            means[model_name] = float(np.mean(scores))
            row[f"{model_name}_mean_auc_roc"] = means[model_name]
            row[f"{model_name}_std_auc_roc"] = sample_std(scores)
            for split, score in zip(model_rows["split_number"], scores):
                row[f"split_{int(split)}_{model_name}_auc_roc"] = float(score)

        for first_model, second_model in itertools.combinations(MODEL_ORDER, 2):
            row[f"{first_model}_minus_{second_model}_mean_auc_roc"] = (
                means[first_model] - means[second_model]
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["comparison_depth", "did"])


def make_depth_summary(per_dataset):
    rows = []
    for depth, group in per_dataset.groupby("comparison_depth", sort=True):
        row = {
            "comparison_depth": int(depth),
            "recurrent_model_effective_transformer_layers": int(depth),
            "tabpfn_effective_transformer_layers": MAX_DEPTH,
            "dataset_count": int(group["did"].nunique()),
            "splits_per_dataset": int(group["split_count"].iloc[0]),
        }

        for model_name in MODEL_ORDER:
            values = group[f"{model_name}_mean_auc_roc"]
            mean, std, ci_low, ci_high = mean_ci95(values)
            row[f"{model_name}_average_auc_roc"] = mean
            row[f"{model_name}_std_auc_roc_across_datasets"] = std
            row[f"{model_name}_ci95_low"] = ci_low
            row[f"{model_name}_ci95_high"] = ci_high

        for first_model, second_model in itertools.combinations(MODEL_ORDER, 2):
            prefix = f"{first_model}_minus_{second_model}"
            differences = group[f"{prefix}_mean_auc_roc"]
            mean, std, ci_low, ci_high = mean_ci95(differences)
            row[f"{prefix}_average_auc_roc"] = mean
            row[f"{prefix}_std_auc_roc_across_datasets"] = std
            row[f"{prefix}_ci95_low"] = ci_low
            row[f"{prefix}_ci95_high"] = ci_high
            row[f"{first_model}_wins_vs_{second_model}"] = int(
                (differences > 0).sum()
            )
            row[f"{first_model}_ties_vs_{second_model}"] = int(
                (differences == 0).sum()
            )
            row[f"{first_model}_losses_vs_{second_model}"] = int(
                (differences < 0).sum()
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.depths = list(dict.fromkeys(args.depths))
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

    benchmark.MODELS = specs
    models = {
        model_name: benchmark.load_model(model_name, args.device)
        for model_name in MODEL_ORDER
    }
    for model_name, (_, config) in models.items():
        validate_config(model_name, config)

    for path in (args.raw_csv, args.per_dataset_csv, args.summary_csv):
        ensure_parent_directory(path)

    raw = evaluate_all(models, specs, datasets, args)
    per_dataset = make_per_dataset(raw)
    summary = make_depth_summary(per_dataset)

    raw.to_csv(args.raw_csv, index=False)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    print("\nMacro-average AUC-ROC by recurrent depth:")
    columns = ["comparison_depth"] + [
        f"{model_name}_average_auc_roc" for model_name in MODEL_ORDER
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nSaved split-level scores to {args.raw_csv}")
    print(f"Saved per-dataset/depth results to {args.per_dataset_csv}")
    print(f"Saved depth summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
