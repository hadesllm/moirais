import pandas as pd
import pytest

from moirais.causal import (
    effective_sample_size,
    run_ebac_selection_ipw_analysis,
    run_propensity_ipw_analysis,
)


def test_run_propensity_ipw_analysis_returns_expected_tables():
    frame = pd.DataFrame(
        {
            "cannabis_any_use": [1, 1, 0, 0, 1, 0],
            "heavy_drinking_30d": [1, 0, 0, 1, 1, 0],
            "age_group": [1, 2, 1, 2, 3, 4],
            "gender": [0, 1, 0, 1, 0, 1],
            "province_region": [0, 1, 0, 1, 0, 1],
            "mental_health": [1, 2, 2, 3, 4, 4],
            "physical_health": [1, 2, 2, 3, 4, 4],
            "weight": [1.0, 1.2, 0.8, 1.1, 1.3, 0.9],
        }
    )

    result = run_propensity_ipw_analysis(frame)

    assert "ipw_results" in result
    assert list(result["ipw_results"]["estimand"]) == ["ATE"]
    assert "ess_ipw_trimmed" in set(result["diagnostics"]["metric"])


@pytest.mark.filterwarnings(
    "ignore:Perfect separation or prediction detected, parameter may not be identified:statsmodels.genmod.generalized_linear_model.PerfectSeparationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide:RuntimeWarning"
)
def test_run_ebac_selection_ipw_analysis_returns_key_outputs():
    frame = pd.DataFrame(
        {
            "weight": [1.0, 1.2, 1.1, 0.9, 1.3, 0.8, 1.05, 0.95],
            "alcohol_past12m": [1, 1, 1, 1, 1, 1, 1, 1],
            "ebac_tot": [0.02, 0.11, 0.08, None, 0.15, 0.07, None, 0.06],
            "ebac_legal": [0, 1, 0, 0, 1, 0, 1, 0],
            "cannabis_any_use": [1, 1, 0, 0, 1, 0, 0, 1],
            "age_group": [1, 2, 1, 2, 3, 4, 2, 3],
            "gender": [0, 1, 0, 1, 0, 1, 1, 0],
            "province_region": [0, 1, 0, 1, 0, 1, 2, 2],
            "mental_health": [1, 2, 2, 3, 4, 4, 3, 2],
            "physical_health": [1, 2, 2, 3, 4, 4, 2, 3],
            "heavy_drinking_30d": [0, 1, 0, 1, 1, 0, 0, 1],
        }
    )

    result = run_ebac_selection_ipw_analysis(frame)

    assert "ebac_final_ipw_or" in result
    assert "ebac_legal_or_cannabis" in set(result["ebac_final_ipw_comparison"]["metric"])


def test_effective_sample_size_positive():
    assert effective_sample_size(pd.Series([1.0, 2.0, 3.0])) > 0
