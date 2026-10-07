"""Compare elastic and standard looped 3x6 models at depths 1 through 6.

At depth d, each model applies its shared ordered three-layer core d times,
for 3*d effective Transformer-layer applications.  Every score is first
averaged over the requested splits within a dataset, then macro-averaged over
datasets.  The default is splits 1-5 on the project's 30 OpenML datasets.

The script writes a per-dataset/depth CSV and a compact depth-average CSV.
"""

import argparse
import os

import numpy as np
import pandas as pd

import evaluation_elastic_vs_looped_3x6_vs_tabpfn_openml as comparison


MAX_DEPTH = 6
MODEL_ORDER = (comparison.ELASTIC_MODEL, comparison.LOOPED_MODEL)


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
        help="Core-stack repetitions to evaluate (default: 1 2 3 4 5 6).",
    )
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs", "elastic_vs_looped_3x6_depths_per_dataset.csv"
        ),
    )
    parser.add_argument(
        "--average-csv",
        default=os.path.join(
            "result_csvs", "elastic_vs_looped_3x6_depths_average.csv"
        ),
    )
    return parser.parse_args()


def evaluate_dataset_depth(models, specs, did, dataset_list, depth, args):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    scores = {model_name: [] for model_name in MODEL_ORDER}

    for split_number in args.splits:
        for model_name in MODEL_ORDER:
            model, config = models[model_name]
            score = comparison.evaluate_one(
                model,
                config,
                model_name,
                specs,
                dataset_list,
                args.device,
                split_number,
                args.max_time,
                num_loops=depth,
            )
            scores[model_name].append(score)
            print(
                f"model={model_name} depth={depth} did={did} "
                f"split={split_number} auc_roc={score:.6f}",
                flush=True,
            )

    elastic_mean = float(np.mean(scores[comparison.ELASTIC_MODEL]))
    looped_mean = float(np.mean(scores[comparison.LOOPED_MODEL]))
    row = {
        "depth": depth,
        "num_loops": depth,
        "block_configuration": f"({depth},{depth},{depth})",
        "effective_transformer_layers": comparison.CORE_LAYERS * depth,
        "did": did,
        "dataset_name": dataset_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "split_count": len(args.splits),
        "elastic_looped_3x6_mean_auc_roc": elastic_mean,
        "elastic_looped_3x6_std_auc_roc": comparison.sample_std(
            scores[comparison.ELASTIC_MODEL]
        ),
        "looped_3x6_mean_auc_roc": looped_mean,
        "looped_3x6_std_auc_roc": comparison.sample_std(
            scores[comparison.LOOPED_MODEL]
        ),
        "elastic_minus_looped_auc_roc": elastic_mean - looped_mean,
    }
    for split_number, elastic_score, looped_score in zip(
        args.splits,
        scores[comparison.ELASTIC_MODEL],
        scores[comparison.LOOPED_MODEL],
    ):
        row[f"split_{split_number}_elastic_looped_3x6_auc_roc"] = elastic_score
        row[f"split_{split_number}_looped_3x6_auc_roc"] = looped_score
    return row


def aggregate_by_depth(per_dataset):
    rows = []
    for depth, depth_frame in per_dataset.groupby("depth", sort=True):
        elastic = depth_frame["elastic_looped_3x6_mean_auc_roc"]
        looped = depth_frame["looped_3x6_mean_auc_roc"]
        difference = depth_frame["elastic_minus_looped_auc_roc"]
        rows.append(
            {
                "depth": int(depth),
                "num_loops": int(depth),
                "block_configuration": f"({depth},{depth},{depth})",
                "effective_transformer_layers": (
                    comparison.CORE_LAYERS * int(depth)
                ),
                "dataset_count": int(depth_frame["did"].nunique()),
                "splits_per_dataset": int(depth_frame["split_count"].iloc[0]),
                "elastic_looped_3x6_average_auc_roc": float(elastic.mean()),
                "elastic_looped_3x6_std_auc_roc_across_datasets": float(
                    elastic.std(ddof=1)
                ),
                "looped_3x6_average_auc_roc": float(looped.mean()),
                "looped_3x6_std_auc_roc_across_datasets": float(
                    looped.std(ddof=1)
                ),
                "elastic_minus_looped_average_auc_roc": float(
                    difference.mean()
                ),
                "elastic_dataset_wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "looped_dataset_wins": int((difference < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.depths = list(dict.fromkeys(args.depths))
    specs, models, datasets = comparison.initialize(args, include_tabpfn=False)

    rows = []
    for depth in args.depths:
        for did, dataset_list in datasets.items():
            rows.append(
                evaluate_dataset_depth(
                    models, specs, did, dataset_list, depth, args
                )
            )

    per_dataset = pd.DataFrame(rows).sort_values(["depth", "did"])
    average = aggregate_by_depth(per_dataset)

    comparison.ensure_parent_directory(args.per_dataset_csv)
    comparison.ensure_parent_directory(args.average_csv)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    average.to_csv(args.average_csv, index=False)

    print("\nMacro-average AUC-ROC by depth:")
    print(average.to_string(index=False))
    print(f"\nSaved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved depth averages to {args.average_csv}")


if __name__ == "__main__":
    main()
