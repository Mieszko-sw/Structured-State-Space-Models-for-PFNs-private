"""Evaluate fixed looped 3xD models, elastic 3x6 depths, and TabPFN.

The default benchmark is the project's 30-dataset OpenML-CC18 subset.  Every
operating point is evaluated on deterministic splits 1--5 with AUC-ROC:

* fixed looped 3x2, 3x3, 3x4, 3x5, and 3x6 at their trained depths;
* elastic looped 3x6 at inference depths 1, 2, 3, 4, 5, and 6;
* the original 12-layer TabPFN once per dataset/split.

Scores are first averaged across splits within each dataset and are then
macro-averaged across datasets, giving every dataset equal weight.  The script
writes split-level, per-dataset, and macro-average CSV files.  Elastic rows also
include same-depth comparisons against the corresponding fixed model whenever
one exists.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_elastic_vs_looped_3x6_vs_tabpfn_openml as common


CORE_LAYERS = 3
TABPFN_EFFECTIVE_LAYERS = 12

FIXED_DEPTHS = {
    "looped_3x2": 2,
    "looped_3x3": 3,
    "looped_3x4": 4,
    "looped_3x5": 5,
    "looped_3x6": 6,
}
ELASTIC_MODEL = "elastic_looped_3x6"
TABPFN_MODEL = "tabpfn"
MODEL_ORDER = tuple(FIXED_DEPTHS) + (ELASTIC_MODEL, TABPFN_MODEL)

DEFAULT_CHECKPOINTS = {
    "looped_3x2": (
        "tabpfn/models_diff/"
        "callback_new_looped_transformer_3physical_core2x_latest.cpkt"
    ),
    "looped_3x3": (
        "tabpfn/models_diff/"
        "callback_new_looped_transformer_3physical_core3x_latest.cpkt"
    ),
    "looped_3x4": (
        "tabpfn/models_diff/"
        "callback_new_looped_transformer_3physical_core4x_latest.cpkt"
    ),
    "looped_3x5": (
        "tabpfn/models_diff/"
        "callback_new_looped_transformer_3physical_core5x_latest.cpkt"
    ),
    "looped_3x6": (
        "tabpfn/models_diff/"
        "callback_new_looped_transformer_3physical_core6x_latest.cpkt"
    ),
    ELASTIC_MODEL: (
        "tabpfn/models_diff/"
        "callback_elastic_looped_transformer_3physical_core6x_latest.cpkt"
    ),
    TABPFN_MODEL: "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
}

DEFAULT_RAW_CSV = os.path.join(
    "result_csvs", "all_looped_3x_elastic_depths_tabpfn_openml30_raw.csv"
)
DEFAULT_PER_DATASET_CSV = os.path.join(
    "result_csvs",
    "all_looped_3x_elastic_depths_tabpfn_openml30_per_dataset.csv",
)
DEFAULT_AVERAGE_CSV = os.path.join(
    "result_csvs", "all_looped_3x_elastic_depths_tabpfn_openml30_average.csv"
)

RAW_COLUMNS = [
    "operating_point",
    "model",
    "model_family",
    "training_max_depth",
    "inference_depth",
    "effective_transformer_layers",
    "checkpoint",
    "did",
    "dataset_name",
    "num_samples",
    "num_features",
    "split_number",
    "auc_roc",
]
RESULT_KEY = ["operating_point", "did", "split_number"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs (default: the project's OpenML-30 list).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        choices=common.benchmark.SPLIT_NUMBERS,
        default=common.benchmark.SPLIT_NUMBERS,
        help="Splits averaged within each dataset (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--elastic-depths",
        nargs="+",
        type=int,
        choices=range(1, 7),
        default=list(range(1, 7)),
        metavar="N",
        help="Elastic 3x6 inference depths (default: 1 2 3 4 5 6).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-time", type=int, default=300)
    for depth in FIXED_DEPTHS.values():
        parser.add_argument(
            f"--looped-3x{depth}-checkpoint",
            default=DEFAULT_CHECKPOINTS[f"looped_3x{depth}"],
        )
    parser.add_argument(
        "--elastic-checkpoint", default=DEFAULT_CHECKPOINTS[ELASTIC_MODEL]
    )
    parser.add_argument(
        "--tabpfn-checkpoint", default=DEFAULT_CHECKPOINTS[TABPFN_MODEL]
    )
    parser.add_argument("--raw-csv", default=DEFAULT_RAW_CSV)
    parser.add_argument("--per-dataset-csv", default=DEFAULT_PER_DATASET_CSV)
    parser.add_argument("--average-csv", default=DEFAULT_AVERAGE_CSV)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed operating-point/dataset/split rows from --raw-csv.",
    )
    return parser.parse_args()


def checkpoint_arguments(args):
    paths = {
        f"looped_3x{depth}": getattr(args, f"looped_3x{depth}_checkpoint")
        for depth in FIXED_DEPTHS.values()
    }
    paths[ELASTIC_MODEL] = args.elastic_checkpoint
    paths[TABPFN_MODEL] = args.tabpfn_checkpoint
    return paths


def build_model_specs(args):
    paths = checkpoint_arguments(args)
    return {
        model_name: {
            "path": paths[model_name],
            "loader_type": (
                "tabpfn" if model_name == TABPFN_MODEL else "looped_transformer"
            ),
            "method_name": "transformer",
        }
        for model_name in MODEL_ORDER
    }


def operating_points(elastic_depths):
    points = []
    for model_name, depth in FIXED_DEPTHS.items():
        points.append(
            {
                "operating_point": model_name,
                "model": model_name,
                "model_family": "fixed_looped",
                "training_max_depth": depth,
                "inference_depth": depth,
                "effective_transformer_layers": CORE_LAYERS * depth,
            }
        )
    for depth in elastic_depths:
        points.append(
            {
                "operating_point": f"{ELASTIC_MODEL}_depth_{depth}",
                "model": ELASTIC_MODEL,
                "model_family": "elastic_looped",
                "training_max_depth": 6,
                "inference_depth": depth,
                "effective_transformer_layers": CORE_LAYERS * depth,
            }
        )
    points.append(
        {
            "operating_point": TABPFN_MODEL,
            "model": TABPFN_MODEL,
            "model_family": "tabpfn",
            "training_max_depth": np.nan,
            "inference_depth": np.nan,
            "effective_transformer_layers": TABPFN_EFFECTIVE_LAYERS,
        }
    )
    return points


def validate_config(model_name, config):
    expected_layers = (
        TABPFN_EFFECTIVE_LAYERS
        if model_name == TABPFN_MODEL
        else CORE_LAYERS * (6 if model_name == ELASTIC_MODEL else FIXED_DEPTHS[model_name])
    )
    if int(config["nlayers"]) != expected_layers:
        raise ValueError(
            f"{model_name}: expected nlayers={expected_layers}, got {config['nlayers']}"
        )

    if model_name == TABPFN_MODEL:
        return
    expected_loop_config = {
        "looped_warmup_layers": 0,
        "looped_core_layers": CORE_LAYERS,
        "looped_exit_layers": 0,
        "looped_core_repeat_pattern": None,
    }
    for key, expected in expected_loop_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{model_name}: expected {key}={expected!r}, got {config.get(key)!r}"
            )


def initialize(args):
    args.splits = list(dict.fromkeys(args.splits))
    args.elastic_depths = list(dict.fromkeys(args.elastic_depths))
    args.device = common.benchmark.normalize_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    specs = build_model_specs(args)
    for model_name, spec in specs.items():
        if not os.path.isfile(spec["path"]):
            raise FileNotFoundError(
                f"Checkpoint for {model_name} does not exist: {spec['path']}"
            )

    eval_helper = common.EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = common.benchmark.prepare_datasets(dids)
    if len(datasets) != len(dids):
        raise RuntimeError(
            f"Requested {len(dids)} datasets but only {len(datasets)} passed preparation."
        )

    common.benchmark.MODELS = specs
    models = {
        model_name: common.benchmark.load_model(model_name, args.device)
        for model_name in MODEL_ORDER
    }
    for model_name, (_, config) in models.items():
        validate_config(model_name, config)
    return specs, models, datasets


def load_existing_raw(args, points, datasets):
    if not args.resume or not os.path.isfile(args.raw_csv):
        return pd.DataFrame(columns=RAW_COLUMNS)
    raw = pd.read_csv(args.raw_csv)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(
            f"Cannot resume from {args.raw_csv}; missing columns: {sorted(missing)}"
        )
    selected_points = {point["operating_point"] for point in points}
    selected_dids = {int(did) for did in datasets}
    selected_splits = {int(split) for split in args.splits}
    raw = raw.loc[
        raw["operating_point"].isin(selected_points)
        & raw["did"].isin(selected_dids)
        & raw["split_number"].isin(selected_splits),
        RAW_COLUMNS,
    ].drop_duplicates(RESULT_KEY, keep="last")
    print(f"Loaded {len(raw)} completed split-level rows from {args.raw_csv}")
    return raw


def completed_keys(raw):
    return {
        (str(row.operating_point), int(row.did), int(row.split_number))
        for row in raw.itertuples(index=False)
    }


def write_raw(rows, path):
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if not raw.empty:
        raw = raw.drop_duplicates(RESULT_KEY, keep="last").sort_values(RESULT_KEY)
    raw.to_csv(path, index=False)
    return raw


def evaluate_all(models, specs, datasets, points, args, existing_raw):
    rows = existing_raw.to_dict("records")
    done = completed_keys(existing_raw)

    for did, dataset_list in datasets.items():
        dataset_name, x, _, _, _, _ = dataset_list[0]
        for point in points:
            model_name = point["model"]
            model, config = models[model_name]
            for split_number in args.splits:
                key = (point["operating_point"], int(did), int(split_number))
                if key in done:
                    continue
                depth = point["inference_depth"]
                num_loops = None if pd.isna(depth) else int(depth)
                score = common.evaluate_one(
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
                row = {
                    **point,
                    "checkpoint": specs[model_name]["path"],
                    "did": int(did),
                    "dataset_name": dataset_name,
                    "num_samples": int(x.shape[0]),
                    "num_features": int(x.shape[1]),
                    "split_number": int(split_number),
                    "auc_roc": float(score),
                }
                rows.append(row)
                done.add(key)
                depth_text = "fixed" if num_loops is None else str(num_loops)
                print(
                    f"model={model_name} depth={depth_text} did={did} "
                    f"split={split_number} auc_roc={score:.6f}",
                    flush=True,
                )

            # Persist after each operating point on every dataset, making the
            # long evaluation safely resumable without repeating five splits.
            write_raw(rows, args.raw_csv)
    return write_raw(rows, args.raw_csv)


def add_reference_comparisons(per_dataset):
    tabpfn = (
        per_dataset.loc[
            per_dataset["operating_point"] == TABPFN_MODEL,
            ["did", "mean_auc_roc"],
        ]
        .rename(columns={"mean_auc_roc": "tabpfn_mean_auc_roc"})
        .drop_duplicates("did")
    )
    result = per_dataset.merge(tabpfn, on="did", how="left", validate="many_to_one")
    result["model_minus_tabpfn_mean_auc_roc"] = (
        result["mean_auc_roc"] - result["tabpfn_mean_auc_roc"]
    )

    fixed = per_dataset.loc[
        per_dataset["model_family"] == "fixed_looped",
        ["did", "inference_depth", "operating_point", "mean_auc_roc"],
    ].rename(
        columns={
            "operating_point": "matched_fixed_model",
            "mean_auc_roc": "matched_fixed_mean_auc_roc",
        }
    )
    result = result.merge(
        fixed,
        on=["did", "inference_depth"],
        how="left",
        validate="many_to_one",
    )
    is_elastic = result["model_family"] == "elastic_looped"
    result.loc[~is_elastic, "matched_fixed_model"] = np.nan
    result.loc[~is_elastic, "matched_fixed_mean_auc_roc"] = np.nan
    result["elastic_minus_matched_fixed_mean_auc_roc"] = np.where(
        is_elastic,
        result["mean_auc_roc"] - result["matched_fixed_mean_auc_roc"],
        np.nan,
    )
    return result


def make_per_dataset(raw, expected_splits):
    group_columns = [
        "operating_point",
        "model",
        "model_family",
        "training_max_depth",
        "inference_depth",
        "effective_transformer_layers",
        "checkpoint",
        "did",
        "dataset_name",
        "num_samples",
        "num_features",
    ]
    # dropna=False is required to retain TabPFN's intentionally undefined depths.
    per_dataset = (
        raw.groupby(group_columns, sort=False, dropna=False)
        .agg(
            split_count=("split_number", "nunique"),
            mean_auc_roc=("auc_roc", "mean"),
            std_auc_roc=("auc_roc", "std"),
        )
        .reset_index()
    )
    incomplete = per_dataset[per_dataset["split_count"] != expected_splits]
    if not incomplete.empty:
        names = ", ".join(
            f"{row.operating_point}/DID{row.did}:{row.split_count}"
            for row in incomplete.itertuples(index=False)
        )
        raise RuntimeError(f"Incomplete dataset operating points: {names}")
    return add_reference_comparisons(per_dataset).sort_values(
        ["model_family", "training_max_depth", "inference_depth", "did"],
        na_position="last",
    )


def make_average(per_dataset, expected_splits):
    group_columns = [
        "operating_point",
        "model",
        "model_family",
        "training_max_depth",
        "inference_depth",
        "effective_transformer_layers",
        "checkpoint",
    ]
    rows = []
    for keys, frame in per_dataset.groupby(group_columns, sort=False, dropna=False):
        identity = dict(zip(group_columns, keys))
        values = frame["mean_auc_roc"]
        vs_tabpfn = frame["model_minus_tabpfn_mean_auc_roc"]
        matched = frame["elastic_minus_matched_fixed_mean_auc_roc"].dropna()
        row = {
            **identity,
            "dataset_count": int(frame["did"].nunique()),
            "splits_per_dataset": expected_splits,
            "average_auc_roc": float(values.mean()),
            "std_auc_roc_across_datasets": float(values.std(ddof=1)),
            "sem_auc_roc_across_datasets": float(
                values.std(ddof=1) / np.sqrt(len(values))
            ),
            "tabpfn_average_auc_roc": float(frame["tabpfn_mean_auc_roc"].mean()),
            "average_auc_roc_minus_tabpfn": float(vs_tabpfn.mean()),
            "dataset_wins_vs_tabpfn": int((vs_tabpfn > 0).sum()),
            "dataset_ties_vs_tabpfn": int((vs_tabpfn == 0).sum()),
            "dataset_losses_vs_tabpfn": int((vs_tabpfn < 0).sum()),
            "matched_fixed_model": (
                frame["matched_fixed_model"].dropna().iloc[0]
                if frame["matched_fixed_model"].notna().any()
                else np.nan
            ),
            "matched_fixed_average_auc_roc": (
                float(frame["matched_fixed_mean_auc_roc"].mean())
                if frame["matched_fixed_mean_auc_roc"].notna().any()
                else np.nan
            ),
            "elastic_minus_matched_fixed_average_auc_roc": (
                float(matched.mean()) if not matched.empty else np.nan
            ),
            "elastic_dataset_wins_vs_matched_fixed": (
                int((matched > 0).sum()) if not matched.empty else np.nan
            ),
            "elastic_dataset_ties_vs_matched_fixed": (
                int((matched == 0).sum()) if not matched.empty else np.nan
            ),
            "elastic_dataset_losses_vs_matched_fixed": (
                int((matched < 0).sum()) if not matched.empty else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["model_family", "training_max_depth", "inference_depth"],
        na_position="last",
    )


def main():
    args = parse_args()
    for path in (args.raw_csv, args.per_dataset_csv, args.average_csv):
        common.ensure_parent_directory(path)

    specs, models, datasets = initialize(args)
    points = operating_points(args.elastic_depths)
    existing_raw = load_existing_raw(args, points, datasets)
    raw = evaluate_all(models, specs, datasets, points, args, existing_raw)
    per_dataset = make_per_dataset(raw, len(args.splits))
    average = make_average(per_dataset, len(args.splits))

    per_dataset.to_csv(args.per_dataset_csv, index=False)
    average.to_csv(args.average_csv, index=False)

    print("\nMacro-average AUC-ROC across OpenML datasets:")
    display_columns = [
        "operating_point",
        "inference_depth",
        "effective_transformer_layers",
        "average_auc_roc",
        "average_auc_roc_minus_tabpfn",
        "elastic_minus_matched_fixed_average_auc_roc",
    ]
    print(average[display_columns].to_string(index=False))
    print(f"\nSaved split-level results to {args.raw_csv}")
    print(f"Saved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved macro-average results to {args.average_csv}")


if __name__ == "__main__":
    main()
