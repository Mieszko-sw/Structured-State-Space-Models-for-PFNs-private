"""Evaluate all fixed looped 3x models at every valid depth on OpenML-30.

Fixed looped 3xN is evaluated with 1 through N applications of its shared
three-layer core. Thus 3x1 is evaluated at 3 effective Transformer layers,
3x2 at 3 and 6 layers, and so on through 3x6 at 3, 6, ..., 18 layers.
Elastic looped 3x6 is also evaluated at depths 1 through 6, while the original
12-layer TabPFN is evaluated once per dataset/split as the reference.

Scores use deterministic splits 1--5 and AUC-ROC. They are averaged first
across splits within each OpenML-30 dataset and then equally across datasets.
Split-level, per-dataset, and macro-average CSV files are written, and a run
can be continued safely with --resume.
"""

import evaluation_all_looped_3x_models_elastic_depths_tabpfn_openml30 as base


LOOPED_3X1 = "looped_3x1"

base.__doc__ = __doc__
base.FIXED_DEPTHS = {
    "looped_3x1": 1,
    "looped_3x2": 2,
    "looped_3x3": 3,
    "looped_3x4": 4,
    "looped_3x5": 5,
    "looped_3x6": 6,
}
base.MODEL_ORDER = tuple(base.FIXED_DEPTHS) + (
    base.ELASTIC_MODEL,
    base.TABPFN_MODEL,
)
base.DEFAULT_CHECKPOINTS[LOOPED_3X1] = (
    "tabpfn/models_diff/"
    "callback_new_looped_transformer_3physical_core1x_latest.cpkt"
)
base.DEFAULT_RAW_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x_all_depths_elastic_tabpfn_openml30_raw.csv",
)
base.DEFAULT_PER_DATASET_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x_all_depths_elastic_tabpfn_openml30_per_dataset.csv",
)
base.DEFAULT_AVERAGE_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x_all_depths_elastic_tabpfn_openml30_average.csv",
)


def operating_points(elastic_depths):
    """Return every fixed-model depth, every requested elastic depth, and TabPFN."""
    points = []
    for model_name, training_max_depth in base.FIXED_DEPTHS.items():
        for inference_depth in range(1, training_max_depth + 1):
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
                "training_max_depth": 6,
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


def add_reference_comparisons(per_dataset):
    """Add TabPFN and elastic-versus-separately-trained-depth comparisons."""
    tabpfn = (
        per_dataset.loc[
            per_dataset["operating_point"] == base.TABPFN_MODEL,
            ["did", "mean_auc_roc"],
        ]
        .rename(columns={"mean_auc_roc": "tabpfn_mean_auc_roc"})
        .drop_duplicates("did")
    )
    result = per_dataset.merge(
        tabpfn,
        on="did",
        how="left",
        validate="many_to_one",
    )
    result["model_minus_tabpfn_mean_auc_roc"] = (
        result["mean_auc_roc"] - result["tabpfn_mean_auc_roc"]
    )

    # At elastic depth d, use the separately trained 3xd checkpoint at its
    # full depth as the unambiguous same-compute fixed reference.
    is_full_depth_fixed = (
        (per_dataset["model_family"] == "fixed_looped")
        & (
            per_dataset["training_max_depth"]
            == per_dataset["inference_depth"]
        )
    )
    fixed = per_dataset.loc[
        is_full_depth_fixed,
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
    result.loc[~is_elastic, "matched_fixed_model"] = base.np.nan
    result.loc[~is_elastic, "matched_fixed_mean_auc_roc"] = base.np.nan
    result["elastic_minus_matched_fixed_mean_auc_roc"] = base.np.where(
        is_elastic,
        result["mean_auc_roc"] - result["matched_fixed_mean_auc_roc"],
        base.np.nan,
    )
    return result


# Reuse the checkpoint loading, validation, evaluation, resumable persistence,
# and averaging implementation from the established OpenML-30 evaluator.
base.operating_points = operating_points
base.add_reference_comparisons = add_reference_comparisons


if __name__ == "__main__":
    base.main()
