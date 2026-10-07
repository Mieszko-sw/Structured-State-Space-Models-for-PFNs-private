"""Evaluate every looped 3x checkpoint at inference depths 1--6 on OpenML-30.

Each fixed checkpoint (looped 3x1, 3x2, ..., 3x6) is evaluated with one
through six applications of its shared three-layer core. This deliberately
includes extrapolation beyond the depth used to train the shallower models;
for example, the 3x1 checkpoint is tested at 3, 6, 9, 12, 15, and 18 effective
Transformer-layer applications. Elastic looped 3x6 is evaluated at the same
six depths, and the original 12-layer TabPFN is used as a fixed reference.

The default benchmark uses deterministic splits 1--5 of the project's
OpenML-30 subset and AUC-ROC. Results are averaged first across splits within
each dataset and then equally across datasets. The macro-average CSV also
reports changes from depth 1 and from the immediately preceding depth.
"""

import evaluation_all_looped_3x_models_all_depths_elastic_tabpfn_openml30 as valid_depths


base = valid_depths.base
MAX_INFERENCE_DEPTH = 6

base.__doc__ = __doc__
base.DEFAULT_RAW_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x_depths_1_to_6_elastic_tabpfn_openml30_raw.csv",
)
base.DEFAULT_PER_DATASET_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x_depths_1_to_6_elastic_tabpfn_openml30_per_dataset.csv",
)
base.DEFAULT_AVERAGE_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x_depths_1_to_6_elastic_tabpfn_openml30_average.csv",
)


def operating_points(elastic_depths):
    """Return depths 1--6 for every fixed model and selected elastic depths."""
    points = []
    for model_name, training_max_depth in base.FIXED_DEPTHS.items():
        for inference_depth in range(1, MAX_INFERENCE_DEPTH + 1):
            points.append(
                {
                    "operating_point": f"{model_name}_depth_{inference_depth}",
                    "model": model_name,
                    "model_family": "fixed_looped",
                    "training_max_depth": training_max_depth,
                    "inference_depth": inference_depth,
                    "effective_transformer_layers": (
                        base.CORE_LAYERS * inference_depth
                    ),
                }
            )

    for inference_depth in elastic_depths:
        points.append(
            {
                "operating_point": (
                    f"{base.ELASTIC_MODEL}_depth_{inference_depth}"
                ),
                "model": base.ELASTIC_MODEL,
                "model_family": "elastic_looped",
                "training_max_depth": MAX_INFERENCE_DEPTH,
                "inference_depth": inference_depth,
                "effective_transformer_layers": (
                    base.CORE_LAYERS * inference_depth
                ),
            }
        )

    points.append(
        {
            "operating_point": base.TABPFN_MODEL,
            "model": base.TABPFN_MODEL,
            "model_family": "tabpfn",
            "training_max_depth": base.np.nan,
            "inference_depth": base.np.nan,
            "effective_transformer_layers": base.TABPFN_EFFECTIVE_LAYERS,
        }
    )
    return points


_make_average = base.make_average


def make_average(per_dataset, expected_splits):
    """Build macro averages and add within-model depth differences."""
    average = _make_average(per_dataset, expected_splits)
    average["average_auc_roc_minus_depth_1"] = base.np.nan
    average["average_auc_roc_minus_previous_depth"] = base.np.nan

    recurrent_models = tuple(base.FIXED_DEPTHS) + (base.ELASTIC_MODEL,)
    for model_name in recurrent_models:
        model_rows = average.loc[average["model"] == model_name].sort_values(
            "inference_depth"
        )
        scores_by_depth = {
            int(row.inference_depth): float(row.average_auc_roc)
            for row in model_rows.itertuples()
        }
        for row in model_rows.itertuples():
            depth = int(row.inference_depth)
            if 1 in scores_by_depth:
                average.loc[
                    row.Index, "average_auc_roc_minus_depth_1"
                ] = row.average_auc_roc - scores_by_depth[1]
            if depth - 1 in scores_by_depth:
                average.loc[
                    row.Index, "average_auc_roc_minus_previous_depth"
                ] = row.average_auc_roc - scores_by_depth[depth - 1]
    return average


# Keep the established loading, configuration validation, resumable raw-result
# persistence, per-dataset aggregation, and reference-comparison behavior.
base.operating_points = operating_points
base.add_reference_comparisons = valid_depths.add_reference_comparisons
base.make_average = make_average


if __name__ == "__main__":
    base.main()
