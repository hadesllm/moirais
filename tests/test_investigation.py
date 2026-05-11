import pandas as pd

from morie.investigation import (
    compare_nested_logistic_models,
    run_treatment_effects_analysis,
    run_weighted_logistic_analysis,
)


def _frame():
    return pd.DataFrame(
        {
            "weight": [1.0, 1.2, 1.1, 0.9, 1.3, 0.8, 1.0, 1.1, 0.95, 1.05],
            "alcohol_past12m": [1] * 10,
            "heavy_drinking_30d": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
            "ebac_tot": [0.02, 0.11, 0.08, 0.04, 0.15, 0.07, 0.05, 0.12, 0.03, 0.10],
            "ebac_legal": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            "cannabis_any_use": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            "age_group": [1, 2, 1, 2, 3, 4, 3, 4, 2, 1],
            "gender": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "province_region": [0, 1, 0, 1, 2, 3, 2, 3, 1, 0],
            "mental_health": [1, 2, 2, 3, 4, 4, 3, 2, 2, 3],
            "physical_health": [1, 2, 2, 3, 4, 4, 3, 2, 2, 3],
        }
    )


def test_run_weighted_logistic_analysis_outputs_expected_tables():
    outputs = run_weighted_logistic_analysis(_frame())
    assert "logistic_odds_ratios" in outputs
    assert "logistic_interaction_tests" in outputs
    assert "OR" in outputs["logistic_odds_ratios"].columns


def test_compare_nested_logistic_models_outputs_summary():
    outputs = compare_nested_logistic_models(_frame())
    assert "model_comparison_summary" in outputs
    assert "model_comparison_wald_tests" in outputs
    assert set(outputs["model_comparison_summary"]["model"]) >= {"Model 0", "Model 1", "Model 2", "Model 3", "Model 4"}


def test_run_treatment_effects_analysis_outputs_summary_and_cate():
    outputs = run_treatment_effects_analysis(_frame())
    assert "treatment_effects_summary" in outputs
    assert "cate_subgroup_estimates" in outputs
    assert set(outputs["treatment_effects_summary"]["estimand"]) >= {"ATE", "ATT", "ATC"}
