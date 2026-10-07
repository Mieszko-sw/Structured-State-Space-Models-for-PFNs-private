"""Evaluate elastic 3x6, looped 3x6, and TabPFN on TabArena-15.

This is the complementary TabArena evaluation for the existing
30-dataset OpenML-CC18 benchmark. The defaults are the exhaustive result of
applying the following predeclared filters to TabArena-v0.1:

  * classification task;
  * 2,000 < number of samples <= 15,000;
  * number of input features <= 100;
  * number of classes <= 10.

The lower sample bound extends the regime covered by OpenML-30, whose largest
datasets contain 2,000 samples. DID 46979 (jm1) is the sole post-filter
exclusion because OpenML-30 already contains four datasets from the same
software-defect prediction family (pc1, pc3, pc4, and kc2). The resulting
benchmark therefore contains 15 additional datasets.

The evaluation protocol is deliberately identical to
``evaluation_elastic_vs_looped_3x6_vs_tabpfn_openml.py``: full recurrent depth
six, five deterministic splits, AUC-ROC, split averaging within each dataset,
and an equal-weight macro-average across datasets.

Outputs:
  * result_csvs/elastic_vs_looped_3x6_vs_tabpfn_tabarena15_per_dataset.csv
  * result_csvs/elastic_vs_looped_3x6_vs_tabpfn_tabarena15_average.csv
"""

import argparse
import os

import pandas as pd

import evaluation_elastic_vs_looped_3x6_vs_tabpfn_openml as comparison


# Exhaustive TabArena-v0.1 classification set after applying
#   2,000 < N <= 15,000, d <= 100, and C <= 10,
# followed by the documented task-family exclusion of DID 46979 (jm1).
#
# Comments give the TabArena-v0.1 full-dataset metadata as (N, d, C). The
# entries are ordered by N. Dataset IDs come from OpenML suite 457.
TABARENA_15_DIDS = [
    46940,  # Marketing_Campaign                         (2,240, 26, 2)
    46930,  # hazelnut-spread-contaminant-detection      (2,400, 31, 2)
    46956,  # seismic-bumps                              (2,584, 16, 2)
    46958,  # splice                                     (3,190, 61, 3)
    46960,  # students_dropout_and_academic_success      (4,424, 37, 3)
    46915,  # churn                                      (5,000, 20, 2)
    46950,  # polish_companies_bankruptcy                (5,910, 65, 2)
    46962,  # taiwanese_bankruptcy_prediction            (6,819, 95, 2)
    46969,  # NATICUSdroid                               (7,491, 87, 2)
    46916,  # coil2000_insurance_policies                (9,822, 86, 2)
    46911,  # Bank_Customer_Churn                       (10,000, 11, 2)
    46932,  # heloc                                     (10,459, 24, 2)
    46924,  # E-CommereShippingData                     (10,999, 11, 2)
    46947,  # online_shoppers_intention                 (12,330, 18, 2)
    46937,  # in_vehicle_coupon_recommendation          (12,684, 25, 2)
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    comparison.add_shared_arguments(parser)
    parser.set_defaults(dids=TABARENA_15_DIDS)
    parser._option_string_actions["--dids"].help = (
        "OpenML dataset IDs (default: the complementary TabArena-15 list)."
    )
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_3x6_vs_tabpfn_tabarena15_per_dataset.csv",
        ),
    )
    parser.add_argument(
        "--average-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_3x6_vs_tabpfn_tabarena15_average.csv",
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    specs, models, datasets = comparison.initialize(args, include_tabpfn=True)

    rows = []
    for did, dataset_list in datasets.items():
        print(f"\nEvaluating TabArena dataset DID {did}", flush=True)
        rows.append(
            comparison.evaluate_dataset(
                models, specs, did, dataset_list, args
            )
        )

    per_dataset = pd.DataFrame(rows).sort_values("did")
    average = comparison.aggregate(per_dataset)

    comparison.ensure_parent_directory(args.per_dataset_csv)
    comparison.ensure_parent_directory(args.average_csv)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    average.to_csv(args.average_csv, index=False)

    average_columns = [
        f"{model_name}_average_auc_roc"
        for model_name in comparison.MODEL_ORDER
    ]
    print("\nMacro-average AUC-ROC across TabArena-15 datasets:")
    print(average[average_columns].to_string(index=False))
    print(f"\nSaved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved model averages to {args.average_csv}")


if __name__ == "__main__":
    main()
