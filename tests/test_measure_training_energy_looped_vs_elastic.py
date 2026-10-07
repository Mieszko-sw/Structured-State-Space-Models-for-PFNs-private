import math

import pytest

from measure_training_energy_looped_vs_elastic import (
    JOULES_PER_KWH,
    build_repeat_scenarios,
    build_summary_rows,
    carbon_kg,
    mean_sd_ci95,
)


def make_raw_rows(scale, repeat):
    rows = []
    for depth in range(1, 7):
        rows.append(
            {
                "repeat": repeat,
                "comparison_role": "individual_fixed_model",
                "fixed_depth": depth,
                "shortcut_depth": "",
                "joules_per_microbatch": scale * depth,
            }
        )
    for shortcut_depth in range(1, 6):
        rows.append(
            {
                "repeat": repeat,
                "comparison_role": "elastic_shortcut_operating_point",
                "fixed_depth": "",
                "shortcut_depth": shortcut_depth,
                "joules_per_microbatch": scale * (6 + shortcut_depth),
            }
        )
    return rows


def test_repeat_scenarios_use_balanced_elastic_mean_and_fixed_sum():
    scenarios = build_repeat_scenarios(make_raw_rows(10.0, 1), range(1, 7), range(1, 6))
    values = {row["scenario"]: row["joules_per_microbatch_equivalent"] for row in scenarios}

    assert values["fixed_models_total"] == 210.0
    assert values["elastic_looped_3x6_expected"] == 90.0
    assert values["elastic_reduction_vs_fixed_suite_percent"] == pytest.approx(
        100.0 * (1.0 - 90.0 / 210.0)
    )


def test_summary_extrapolates_energy_and_co2():
    raw_rows = make_raw_rows(10.0, 1) + make_raw_rows(12.0, 2)
    total_microsteps = 1_000
    summary = build_summary_rows(
        raw_rows,
        range(1, 7),
        range(1, 6),
        total_microsteps=total_microsteps,
        carbon_intensity_g_per_kwh=400.0,
        pue=1.2,
    )
    elastic = next(row for row in summary if row["comparison_role"] == "elastic_model")
    comparison = next(row for row in summary if row["comparison_role"] == "comparison")

    expected_joules_per_microbatch = 99.0
    expected_kwh = expected_joules_per_microbatch * total_microsteps / JOULES_PER_KWH
    assert elastic["mean_joules_per_microbatch_equivalent"] == expected_joules_per_microbatch
    assert elastic["estimated_full_training_gpu_kwh"] == expected_kwh
    assert elastic["estimated_full_training_operational_co2e_kg"] == pytest.approx(
        expected_kwh * 1.2 * 400.0 / 1000.0
    )
    assert comparison["mean_elastic_reduction_vs_fixed_suite_percent"] == pytest.approx(
        100.0 * (1.0 - 9.0 / 21.0)
    )


def test_carbon_is_optional_and_confidence_interval_handles_one_value():
    assert carbon_kg(2.0, None, 1.0) is None
    assert carbon_kg(2.0, 500.0, 1.2) == pytest.approx(1.2)
    mean, standard_deviation, low, high = mean_sd_ci95([7.0])
    assert (mean, standard_deviation, low, high) == (7.0, 0.0, 7.0, 7.0)
    assert math.isfinite(mean)
