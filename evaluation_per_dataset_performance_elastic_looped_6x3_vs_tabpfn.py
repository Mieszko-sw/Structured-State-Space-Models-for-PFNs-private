"""Evaluate elastic 6x3, standard looped 6x3, and TabPFN on OpenML.

Here ``6x3`` means six recurrent passes through a shared three-layer core,
giving 18 effective Transformer-layer applications.  By default, every model
is evaluated exactly once on split 1 of each of the project's 30 OpenML
datasets.  This script measures AUC-ROC performance only; it contains no
warm-up, timing, synchronization, or repeated-run logic.

Two CSVs are written:
  * one row per dataset with all three model scores and pairwise differences;
  * one row with equal-weight macro-averages and dataset win counts.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


NUM_LOOPS = 6
CORE_LAYERS = 3
ELASTIC_MODEL = "elastic_looped_6x3"
LOOPED_MODEL = "looped_6x3"
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split-number",
        type=int,
        choices=benchmark.SPLIT_NUMBERS,
        default=1,
        help="Single split evaluated for every dataset (default: 1).",
    )
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs (default: the project's 30-dataset list).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument(
        "--elastic-checkpoint", default=DEFAULT_ELASTIC_CHECKPOINT
    )
    parser.add_argument("--looped-checkpoint", default=DEFAULT_LOOPED_CHECKPOINT)
    parser.add_argument("--tabpfn-checkpoint", default=DEFAULT_TABPFN_CHECKPOINT)
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "performance_elastic_looped_6x3_vs_tabpfn_per_dataset.csv",
        ),
    )
    parser.add_argument(
        "--average-csv",
        default=os.path.join(
            "result_csvs",
            "performance_elastic_looped_6x3_vs_tabpfn_average.csv",
        ),
    )
    return parser.parse_args()


def model_specs(args):
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


def ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def evaluate_once(
    model,
    config,
    model_name,
    specs,
    dataset_list,
    args,
):
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(
            f"{model_name}: expected one evaluation position, got {eval_positions}"
        )

    depth_argument = (
        {} if model_name == TABPFN_MODEL else {"num_loops": NUM_LOOPS}
    )
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
            split_number=args.split_number,
            jrt_prompt=False,
            random_premutation=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            **depth_argument,
        )
    return benchmark.extract_metric(result)


def evaluate_dataset(models, specs, did, dataset_list, args):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    scores = {}
    for model_name in MODEL_ORDER:
        model, config = models[model_name]
        score = evaluate_once(
            model,
            config,
            model_name,
            specs,
            dataset_list,
            args,
        )
        scores[model_name] = score
        print(
            f"model={model_name} did={did} split={args.split_number} "
            f"auc_roc={score:.6f}",
            flush=True,
        )

    return {
        "did": did,
        "dataset_name": dataset_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "split_number": args.split_number,
        "num_loops": NUM_LOOPS,
        "effective_transformer_layers": CORE_LAYERS * NUM_LOOPS,
        "elastic_looped_6x3_auc_roc": scores[ELASTIC_MODEL],
        "looped_6x3_auc_roc": scores[LOOPED_MODEL],
        "tabpfn_auc_roc": scores[TABPFN_MODEL],
        "elastic_minus_looped_auc_roc": (
            scores[ELASTIC_MODEL] - scores[LOOPED_MODEL]
        ),
        "elastic_minus_tabpfn_auc_roc": (
            scores[ELASTIC_MODEL] - scores[TABPFN_MODEL]
        ),
        "looped_minus_tabpfn_auc_roc": (
            scores[LOOPED_MODEL] - scores[TABPFN_MODEL]
        ),
    }


def aggregate(per_dataset, split_number):
    elastic_minus_looped = per_dataset["elastic_minus_looped_auc_roc"]
    elastic_minus_tabpfn = per_dataset["elastic_minus_tabpfn_auc_roc"]
    looped_minus_tabpfn = per_dataset["looped_minus_tabpfn_auc_roc"]
    return pd.DataFrame(
        [
            {
                "dataset_count": int(per_dataset["did"].nunique()),
                "evaluations_per_model_per_dataset": 1,
                "split_number": split_number,
                "num_loops": NUM_LOOPS,
                "effective_transformer_layers": CORE_LAYERS * NUM_LOOPS,
                "elastic_looped_6x3_average_auc_roc": float(
                    per_dataset["elastic_looped_6x3_auc_roc"].mean()
                ),
                "looped_6x3_average_auc_roc": float(
                    per_dataset["looped_6x3_auc_roc"].mean()
                ),
                "tabpfn_average_auc_roc": float(
                    per_dataset["tabpfn_auc_roc"].mean()
                ),
                "elastic_minus_looped_average_auc_roc": float(
                    elastic_minus_looped.mean()
                ),
                "elastic_minus_tabpfn_average_auc_roc": float(
                    elastic_minus_tabpfn.mean()
                ),
                "looped_minus_tabpfn_average_auc_roc": float(
                    looped_minus_tabpfn.mean()
                ),
                "elastic_wins_vs_looped": int(
                    (elastic_minus_looped > 0).sum()
                ),
                "looped_wins_vs_elastic": int(
                    (elastic_minus_looped < 0).sum()
                ),
                "elastic_wins_vs_tabpfn": int(
                    (elastic_minus_tabpfn > 0).sum()
                ),
                "tabpfn_wins_vs_elastic": int(
                    (elastic_minus_tabpfn < 0).sum()
                ),
                "looped_wins_vs_tabpfn": int(
                    (looped_minus_tabpfn > 0).sum()
                ),
                "tabpfn_wins_vs_looped": int(
                    (looped_minus_tabpfn < 0).sum()
                ),
            }
        ]
    )


def main():
    args = parse_args()
    args.device = benchmark.normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    specs = model_specs(args)
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

    rows = [
        evaluate_dataset(models, specs, did, dataset_list, args)
        for did, dataset_list in datasets.items()
    ]
    per_dataset = pd.DataFrame(rows).sort_values("did")
    average = aggregate(per_dataset, args.split_number)

    ensure_parent_directory(args.per_dataset_csv)
    ensure_parent_directory(args.average_csv)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    average.to_csv(args.average_csv, index=False)

    print("\nMacro-average AUC-ROC across datasets:")
    columns = [
        "elastic_looped_6x3_average_auc_roc",
        "looped_6x3_average_auc_roc",
        "tabpfn_average_auc_roc",
    ]
    print(average[columns].to_string(index=False))
    print(f"\nSaved per-dataset scores to {args.per_dataset_csv}")
    print(f"Saved macro-averages to {args.average_csv}")


if __name__ == "__main__":
    main()
