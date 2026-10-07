"""Evaluate fixed looped 3x1--3x6 models, elastic 3x6 depths, and TabPFN.

This is the 3x1-inclusive companion to
``evaluation_all_looped_3x_models_elastic_depths_tabpfn_openml30.py``. It uses
the same OpenML-30 datasets, deterministic splits, metrics, resume behavior,
and aggregation logic while adding the newly trained fixed looped 3x1 model.

Separate default CSV paths are used so running this evaluator does not
overwrite results produced by the original 3x2--3x6 evaluation.
"""

import evaluation_all_looped_3x_models_elastic_depths_tabpfn_openml30 as base


LOOPED_3X1 = "looped_3x1"

# The base evaluator derives its CLI arguments, model specs, validation, and
# operating points from these module globals. Extend them before calling main
# so all behavior stays synchronized with the original evaluator.
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
    "all_looped_3x1_to_3x6_elastic_depths_tabpfn_openml30_raw.csv",
)
base.DEFAULT_PER_DATASET_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x1_to_3x6_elastic_depths_tabpfn_openml30_per_dataset.csv",
)
base.DEFAULT_AVERAGE_CSV = base.os.path.join(
    "result_csvs",
    "all_looped_3x1_to_3x6_elastic_depths_tabpfn_openml30_average.csv",
)


if __name__ == "__main__":
    base.main()
