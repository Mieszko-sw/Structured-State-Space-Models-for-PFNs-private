"""Compare elastic 3x6, standard looped 3x6, and original TabPFN.

The default benchmark is the project's 30-dataset OpenML-CC18 subset and
splits 1-5.  Both looped models use their full six repetitions of a shared
three-layer core (18 effective layer applications).  Scores are averaged
across splits within each dataset and then macro-averaged across datasets so
that every dataset has equal weight.

Outputs:
  * a wide per-dataset CSV with split scores, means, standard deviations, and
    pairwise differences;
  * a one-row macro-average CSV with model averages and dataset win counts.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


FULL_DEPTH = 6
CORE_LAYERS = 3
ELASTIC_MODEL = "elastic_looped_3x6"
LOOPED_MODEL = "looped_3x6"
TABPFN_MODEL = "tabpfn"
MODEL_ORDER = (ELASTIC_MODEL, LOOPED_MODEL, TABPFN_MODEL)

DEFAULT_ELASTIC_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_3physical_core6x_latest.cpkt"
)
DEFAULT_LOOPED_CHECKPOINT = (
    "tabpfn/models_diff/"
    "callback_new_looped_transformer_3physical_core6x_latest.cpkt"
)
DEFAULT_TABPFN_CHECKPOINT = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"


def build_model_specs(args):
    return {
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


def add_shared_arguments(parser):
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
    parser.add_argument(
        "--elastic-checkpoint", default=DEFAULT_ELASTIC_CHECKPOINT
    )
    parser.add_argument("--looped-checkpoint", default=DEFAULT_LOOPED_CHECKPOINT)
    parser.add_argument("--tabpfn-checkpoint", default=DEFAULT_TABPFN_CHECKPOINT)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs", "elastic_vs_looped_3x6_vs_tabpfn_per_dataset.csv"
        ),
    )
    parser.add_argument(
        "--average-csv",
        default=os.path.join(
            "result_csvs", "elastic_vs_looped_3x6_vs_tabpfn_average.csv"
        ),
    )
    return parser.parse_args()


def ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def initialize(args, include_tabpfn=True):
    args.splits = list(dict.fromkeys(args.splits))
    args.device = benchmark.normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    specs = build_model_specs(args)
    selected_models = MODEL_ORDER if include_tabpfn else MODEL_ORDER[:2]
    for model_name in selected_models:
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

    # load_model reads its registry from the benchmark module.
    benchmark.MODELS = specs
    models = {
        model_name: benchmark.load_model(model_name, args.device)
        for model_name in selected_models
    }
    return specs, models, datasets


def evaluate_one(
    model,
    config,
    model_name,
    model_specs,
    dataset_list,
    device,
    split_number,
    max_time,
    num_loops=None,
):
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(
            f"{model_name}: expected one evaluation position, got {eval_positions}"
        )

    depth_argument = {} if num_loops is None else {"num_loops": num_loops}
    with torch.no_grad():
        result = benchmark.evaluate(
            datasets=dataset_list,
            bptt=benchmark.BPTT,
            eval_positions=eval_positions,
            metric_used=benchmark.METRIC_USED,
            model=model,
            device=device,
            method_name=model_specs[model_name]["method_name"],
            max_time=max_time,
            split_number=split_number,
            jrt_prompt=False,
            random_premutation=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            **depth_argument,
        )
    return benchmark.extract_metric(result)


def sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def evaluate_dataset(models, specs, did, dataset_list, args):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    scores = {model_name: [] for model_name in MODEL_ORDER}

    for split_number in args.splits:
        for model_name in MODEL_ORDER:
            model, config = models[model_name]
            num_loops = None if model_name == TABPFN_MODEL else FULL_DEPTH
            score = evaluate_one(
                model,
                config,
                model_name,
                specs,
                dataset_list,
                args.device,
                split_number,
                args.max_time,
                num_loops=num_loops,
            )
            scores[model_name].append(score)
            print(
                f"model={model_name} did={did} split={split_number} "
                f"auc_roc={score:.6f}",
                flush=True,
            )

    means = {
        model_name: float(np.mean(model_scores))
        for model_name, model_scores in scores.items()
    }
    row = {
        "did": did,
        "dataset_name": dataset_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "split_count": len(args.splits),
        "num_loops": FULL_DEPTH,
        "effective_transformer_layers": CORE_LAYERS * FULL_DEPTH,
    }
    for model_name in MODEL_ORDER:
        row[f"{model_name}_mean_auc_roc"] = means[model_name]
        row[f"{model_name}_std_auc_roc"] = sample_std(scores[model_name])
        for split_number, score in zip(args.splits, scores[model_name]):
            row[f"split_{split_number}_{model_name}_auc_roc"] = score

    row["elastic_minus_looped_auc_roc"] = (
        means[ELASTIC_MODEL] - means[LOOPED_MODEL]
    )
    row["elastic_minus_tabpfn_auc_roc"] = (
        means[ELASTIC_MODEL] - means[TABPFN_MODEL]
    )
    row["looped_minus_tabpfn_auc_roc"] = (
        means[LOOPED_MODEL] - means[TABPFN_MODEL]
    )
    return row


def aggregate(per_dataset):
    elastic_minus_looped = per_dataset["elastic_minus_looped_auc_roc"]
    elastic_minus_tabpfn = per_dataset["elastic_minus_tabpfn_auc_roc"]
    looped_minus_tabpfn = per_dataset["looped_minus_tabpfn_auc_roc"]

    row = {
        "dataset_count": int(per_dataset["did"].nunique()),
        "splits_per_dataset": int(per_dataset["split_count"].iloc[0]),
        "num_loops": FULL_DEPTH,
        "effective_transformer_layers": CORE_LAYERS * FULL_DEPTH,
    }
    for model_name in MODEL_ORDER:
        values = per_dataset[f"{model_name}_mean_auc_roc"]
        row[f"{model_name}_average_auc_roc"] = float(values.mean())
        row[f"{model_name}_std_auc_roc_across_datasets"] = float(
            values.std(ddof=1)
        )

    row.update(
        {
            "elastic_minus_looped_average_auc_roc": float(
                elastic_minus_looped.mean()
            ),
            "elastic_minus_tabpfn_average_auc_roc": float(
                elastic_minus_tabpfn.mean()
            ),
            "looped_minus_tabpfn_average_auc_roc": float(
                looped_minus_tabpfn.mean()
            ),
            "elastic_wins_vs_looped": int((elastic_minus_looped > 0).sum()),
            "looped_wins_vs_elastic": int((elastic_minus_looped < 0).sum()),
            "elastic_wins_vs_tabpfn": int((elastic_minus_tabpfn > 0).sum()),
            "tabpfn_wins_vs_elastic": int((elastic_minus_tabpfn < 0).sum()),
            "looped_wins_vs_tabpfn": int((looped_minus_tabpfn > 0).sum()),
            "tabpfn_wins_vs_looped": int((looped_minus_tabpfn < 0).sum()),
        }
    )
    return pd.DataFrame([row])


def main():
    args = parse_args()
    specs, models, datasets = initialize(args, include_tabpfn=True)

    rows = [
        evaluate_dataset(models, specs, did, dataset_list, args)
        for did, dataset_list in datasets.items()
    ]
    per_dataset = pd.DataFrame(rows).sort_values("did")
    average = aggregate(per_dataset)

    ensure_parent_directory(args.per_dataset_csv)
    ensure_parent_directory(args.average_csv)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    average.to_csv(args.average_csv, index=False)

    print("\nMacro-average AUC-ROC across OpenML datasets:")
    average_columns = [
        f"{model_name}_average_auc_roc" for model_name in MODEL_ORDER
    ]
    print(average[average_columns].to_string(index=False))
    print(f"\nSaved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved model averages to {args.average_csv}")


if __name__ == "__main__":
    main()
