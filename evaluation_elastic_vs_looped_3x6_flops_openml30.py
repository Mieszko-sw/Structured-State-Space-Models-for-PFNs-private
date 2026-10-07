"""Profile FLOPs and predictive performance on the OpenML-30 benchmark.

The script evaluates elastic looped 3x6 and standard looped 3x6 at recurrent
depths 1--6, plus the original 12-layer TabPFN once per dataset.  It reuses
the dataset preparation, five deterministic splits, AUC-ROC metric, and model
checkpoints from ``evaluation_elastic_vs_looped_3x6_vs_tabpfn_openml.py``.

FLOPs are collected for one selected split per dataset with PyTorch's
``torch.profiler`` and reused for the other splits, whose tensor shapes are
identical.  The profiler currently estimates FLOPs for supported operators
(principally matrix multiplication and convolution); consequently, output
columns are explicitly named ``profiler_counted_*`` rather than claiming an
exact count of every scalar operation.

Outputs:
  * a split-level raw CSV containing every predictive score;
  * a tidy per-dataset/model/depth CSV containing FLOPs and mean AUC-ROC;
  * a macro-average CSV containing the performance--compute trade-off.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import evaluation_elastic_vs_looped_3x6_vs_tabpfn_openml as comparison
import evaluation_per_dataset_speed_synchronized as benchmark


MAX_DEPTH = 6
TABPFN_EFFECTIVE_LAYERS = 12
LOOPED_MODELS = (comparison.ELASTIC_MODEL, comparison.LOOPED_MODEL)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    comparison.add_shared_arguments(parser)
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        choices=range(1, MAX_DEPTH + 1),
        default=list(range(1, MAX_DEPTH + 1)),
        metavar="N",
        help="Looped-model inference depths (default: 1 2 3 4 5 6).",
    )
    parser.add_argument(
        "--profile-split",
        type=int,
        choices=benchmark.SPLIT_NUMBERS,
        default=1,
        help=(
            "Split on which FLOPs are profiled for each dataset/model/depth "
            "(default: 1). It must also be present in --splits."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help=(
            "Unprofiled warm-up forwards per loaded model before measurement "
            "(default: 1; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_3x6_flops_openml30_raw.csv",
        ),
    )
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_3x6_flops_openml30_per_dataset.csv",
        ),
    )
    parser.add_argument(
        "--average-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_3x6_flops_openml30_average.csv",
        ),
    )
    return parser.parse_args()


def run_specs(depths):
    """Return model/depth operating points; TabPFN is intentionally unique."""
    specs = [(comparison.TABPFN_MODEL, None, TABPFN_EFFECTIVE_LAYERS)]
    for model_name in LOOPED_MODELS:
        specs.extend(
            (model_name, depth, comparison.CORE_LAYERS * depth)
            for depth in depths
        )
    return specs


def effective_evaluation_shape(dataset_list, config):
    """Mirror tabular_evaluation.evaluate's effective balanced split."""
    _, x, _, _, _, _ = dataset_list[0]
    dataset_bptt = min(len(x), benchmark.BPTT)
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(f"Expected one evaluation position, got {eval_positions}")
    configured_position = int(eval_positions[0])
    if 2 * configured_position > dataset_bptt:
        context_examples = int(dataset_bptt * 0.5)
    else:
        context_examples = configured_position
    query_examples = context_examples
    return context_examples, query_examples, 2 * context_examples


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def evaluate_score(
    models,
    model_specs,
    model_name,
    dataset_list,
    args,
    split_number,
    depth,
):
    model, config = models[model_name]
    return comparison.evaluate_one(
        model,
        config,
        model_name,
        model_specs,
        dataset_list,
        args.device,
        split_number,
        args.max_time,
        num_loops=depth,
    )


def profile_score(
    models,
    model_specs,
    model_name,
    dataset_list,
    args,
    split_number,
    depth,
):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    synchronize(args.device)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=True,
    ) as prof:
        score = evaluate_score(
            models,
            model_specs,
            model_name,
            dataset_list,
            args,
            split_number,
            depth,
        )
        synchronize(args.device)

    counted_events = [
        event for event in prof.key_averages() if (event.flops or 0) > 0
    ]
    counted_flops = int(sum(int(event.flops) for event in counted_events))
    counted_operator_types = ";".join(sorted(event.key for event in counted_events))
    return score, counted_flops, counted_operator_types


def warm_up(models, model_specs, datasets, args):
    if args.warmup_runs == 0:
        return
    first_did = next(iter(datasets))
    dataset_list = datasets[first_did]
    maximum_depth = max(args.depths)
    print(
        f"Warming up models on DID {first_did} "
        f"({args.warmup_runs} run(s) each)...",
        flush=True,
    )
    for model_name in comparison.MODEL_ORDER:
        depth = None if model_name == comparison.TABPFN_MODEL else maximum_depth
        for _ in range(args.warmup_runs):
            evaluate_score(
                models,
                model_specs,
                model_name,
                dataset_list,
                args,
                args.profile_split,
                depth,
            )
    synchronize(args.device)


def evaluate_operating_point(
    models,
    model_specs,
    model_name,
    depth,
    effective_layers,
    did,
    dataset_list,
    args,
):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    model, config = models[model_name]
    context_examples, query_examples, sequence_length = (
        effective_evaluation_shape(dataset_list, config)
    )

    raw_rows = []
    scores = []
    counted_flops = None
    counted_operator_types = None
    for split_number in args.splits:
        if split_number == args.profile_split:
            score, counted_flops, counted_operator_types = profile_score(
                models,
                model_specs,
                model_name,
                dataset_list,
                args,
                split_number,
                depth,
            )
            profiled = True
        else:
            score = evaluate_score(
                models,
                model_specs,
                model_name,
                dataset_list,
                args,
                split_number,
                depth,
            )
            profiled = False

        scores.append(score)
        raw_rows.append(
            {
                "model": model_name,
                "num_loops": depth,
                "effective_transformer_layers": effective_layers,
                "did": did,
                "dataset_name": dataset_name,
                "split_number": split_number,
                "auc_roc": score,
                "profiled_for_flops": profiled,
                "profiler_counted_flops": (
                    counted_flops if profiled else np.nan
                ),
            }
        )
        print(
            f"model={model_name} depth={depth} did={did} "
            f"split={split_number} auc_roc={score:.6f}"
            + (
                f" profiler_counted_gflops={counted_flops / 1e9:.6f}"
                if profiled
                else ""
            ),
            flush=True,
        )

    if counted_flops is None:
        raise RuntimeError(
            f"No FLOP profile was captured for {model_name}, DID {did}."
        )
    if counted_flops == 0:
        print(
            f"WARNING: profiler counted zero supported FLOPs for "
            f"{model_name}, DID {did}.",
            flush=True,
        )

    per_dataset_row = {
        "model": model_name,
        "num_loops": depth,
        "effective_transformer_layers": effective_layers,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "did": did,
        "dataset_name": dataset_name,
        "prepared_num_samples": int(x.shape[0]),
        "prepared_num_features": int(x.shape[1]),
        "model_feature_capacity": int(config.get("num_features", x.shape[1])),
        "context_examples": context_examples,
        "query_examples": query_examples,
        "sequence_length": sequence_length,
        "split_count": len(args.splits),
        "profiled_split": args.profile_split,
        "mean_auc_roc": float(np.mean(scores)),
        "std_auc_roc": comparison.sample_std(scores),
        "profiler_counted_flops": counted_flops,
        "profiler_counted_gflops": counted_flops / 1e9,
        "profiler_counted_flops_per_query": counted_flops / query_examples,
        "profiler_counted_operator_types": counted_operator_types,
    }
    return raw_rows, per_dataset_row


def add_tabpfn_comparisons(per_dataset):
    per_dataset = per_dataset.copy()
    tabpfn = per_dataset[
        per_dataset["model"] == comparison.TABPFN_MODEL
    ].set_index("did")
    tabpfn_flops = tabpfn["profiler_counted_flops"]
    tabpfn_auc = tabpfn["mean_auc_roc"]
    per_dataset["relative_flops_vs_tabpfn"] = per_dataset.apply(
        lambda row: row["profiler_counted_flops"] / tabpfn_flops.loc[row["did"]],
        axis=1,
    )
    per_dataset["auc_roc_minus_tabpfn"] = per_dataset.apply(
        lambda row: row["mean_auc_roc"] - tabpfn_auc.loc[row["did"]],
        axis=1,
    )
    return per_dataset


def select_operating_point(per_dataset, model_name, depth):
    selected = per_dataset["model"] == model_name
    if depth is None:
        selected &= per_dataset["num_loops"].isna()
    else:
        selected &= per_dataset["num_loops"] == depth
    return per_dataset[selected]


def aggregate(per_dataset, specs, split_count):
    tabpfn = select_operating_point(
        per_dataset, comparison.TABPFN_MODEL, None
    )
    tabpfn_total_flops = float(tabpfn["profiler_counted_flops"].sum())
    rows = []
    for model_name, depth, effective_layers in specs:
        frame = select_operating_point(per_dataset, model_name, depth)
        differences = frame["auc_roc_minus_tabpfn"]
        one_split_total = float(frame["profiler_counted_flops"].sum())
        rows.append(
            {
                "model": model_name,
                "num_loops": depth,
                "effective_transformer_layers": effective_layers,
                "parameter_count": int(frame["parameter_count"].iloc[0]),
                "dataset_count": int(frame["did"].nunique()),
                "splits_per_dataset": split_count,
                "average_auc_roc": float(frame["mean_auc_roc"].mean()),
                "std_auc_roc_across_datasets": float(
                    frame["mean_auc_roc"].std(ddof=1)
                ),
                "average_profiler_counted_gflops": float(
                    frame["profiler_counted_gflops"].mean()
                ),
                "median_profiler_counted_gflops": float(
                    frame["profiler_counted_gflops"].median()
                ),
                "profiler_counted_gflops_one_split_all_datasets": (
                    one_split_total / 1e9
                ),
                "profiler_counted_gflops_complete_evaluation": (
                    one_split_total * split_count / 1e9
                ),
                "flops_ratio_of_dataset_totals_vs_tabpfn": (
                    one_split_total / tabpfn_total_flops
                ),
                "average_relative_flops_vs_tabpfn": float(
                    frame["relative_flops_vs_tabpfn"].mean()
                ),
                "average_auc_roc_minus_tabpfn": float(differences.mean()),
                "dataset_wins_vs_tabpfn": int((differences > 0).sum()),
                "ties_vs_tabpfn": int((differences == 0).sum()),
                "dataset_losses_vs_tabpfn": int((differences < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_progress(raw_rows, per_dataset_rows, args):
    comparison.ensure_parent_directory(args.raw_csv)
    comparison.ensure_parent_directory(args.per_dataset_csv)
    pd.DataFrame(raw_rows).to_csv(args.raw_csv, index=False)
    if per_dataset_rows:
        current = add_tabpfn_comparisons(pd.DataFrame(per_dataset_rows))
        current.to_csv(args.per_dataset_csv, index=False)


def main():
    args = parse_args()
    args.depths = list(dict.fromkeys(args.depths))
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.profile_split not in args.splits:
        raise ValueError("--profile-split must also be included in --splits.")

    model_specs, models, datasets = comparison.initialize(
        args, include_tabpfn=True
    )
    specs = run_specs(args.depths)
    warm_up(models, model_specs, datasets, args)

    raw_rows = []
    per_dataset_rows = []
    for did, dataset_list in datasets.items():
        print(f"\nProfiling OpenML dataset DID {did}", flush=True)
        for model_name, depth, effective_layers in specs:
            point_raw, point_dataset = evaluate_operating_point(
                models,
                model_specs,
                model_name,
                depth,
                effective_layers,
                did,
                dataset_list,
                args,
            )
            raw_rows.extend(point_raw)
            per_dataset_rows.append(point_dataset)
        # Preserve all fully completed datasets if the long run is interrupted.
        save_progress(raw_rows, per_dataset_rows, args)

    per_dataset = add_tabpfn_comparisons(pd.DataFrame(per_dataset_rows))
    average = aggregate(per_dataset, specs, len(args.splits))

    pd.DataFrame(raw_rows).to_csv(args.raw_csv, index=False)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    comparison.ensure_parent_directory(args.average_csv)
    average.to_csv(args.average_csv, index=False)

    print("\nOpenML-30 performance--compute summary:")
    display_columns = [
        "model",
        "num_loops",
        "effective_transformer_layers",
        "average_auc_roc",
        "average_profiler_counted_gflops",
        "flops_ratio_of_dataset_totals_vs_tabpfn",
        "average_auc_roc_minus_tabpfn",
    ]
    print(average[display_columns].to_string(index=False))
    print(f"\nSaved split-level results to {args.raw_csv}")
    print(f"Saved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved aggregate results to {args.average_csv}")


if __name__ == "__main__":
    main()
