"""Tests for morie.fn.cmp_log — nested logistic model comparison."""

import math

import numpy as np
import pandas as pd
import pytest

from morie.fn.cmp_log import compare_nested_logistic_models


def _make_cpads_frame(rng, n=200):
    """Create a minimal CPADS-compliant synthetic frame with all required columns."""
    age = rng.choice(["18-24", "25-34", "35-44", "45-54"], size=n)
    gender = rng.choice(["male", "female"], size=n)
    province = rng.choice(["ON", "QC", "BC", "AB"], size=n)
    mental = rng.choice(["excellent", "good", "fair", "poor"], size=n)
    physical = rng.choice(["excellent", "good", "fair", "poor"], size=n)
    treatment = rng.integers(0, 2, size=n)
    p_outcome = 0.2 + 0.15 * treatment
    outcome = rng.binomial(1, p_outcome, size=n)
    weight = rng.uniform(0.5, 3.0, size=n)
    return pd.DataFrame({
        "weight": weight,
        "alcohol_past12m": rng.integers(0, 2, size=n),
        "heavy_drinking_30d": outcome,
        "ebac_tot": rng.uniform(0, 0.15, size=n),
        "ebac_legal": rng.integers(0, 2, size=n),
        "cannabis_any_use": treatment,
        "age_group": age,
        "gender": gender,
        "province_region": province,
        "mental_health": mental,
        "physical_health": physical,
    })


@pytest.fixture()
def cmp_data():
    rng = np.random.default_rng(42)
    return _make_cpads_frame(rng, n=200)


def test_returns_comparison_result(cmp_data):
    """compare_nested_logistic_models returns a dict with summary table."""
    result = compare_nested_logistic_models(cmp_data)
    assert isinstance(result, dict)
    assert "model_comparison_summary" in result
    summary = result["model_comparison_summary"]
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 5  # 5 nested models


def test_nested_model_has_fewer_params(cmp_data):
    """Each successive model has at least as many parameters as the previous."""
    result = compare_nested_logistic_models(cmp_data)
    summary = result["model_comparison_summary"]
    n_params = summary["n_parameters"].tolist()
    for i in range(1, len(n_params)):
        assert n_params[i] >= n_params[i - 1], (
            f"Model {i} has fewer params ({n_params[i]}) than Model {i-1} ({n_params[i-1]})"
        )


def test_wald_test_p_values_are_finite(cmp_data):
    """Per-predictor Wald test p-values are all finite numbers."""
    result = compare_nested_logistic_models(cmp_data)
    wald = result["model_comparison_wald_tests"]
    if len(wald) > 0:
        assert wald["p_value"].apply(math.isfinite).all()
        assert wald["F_statistic"].apply(math.isfinite).all()


def test_works_with_synthetic_data(cmp_data):
    """Full pipeline completes on synthetic data and produces all expected keys."""
    result = compare_nested_logistic_models(cmp_data)
    expected_keys = {
        "analysis_frame",
        "model_comparison_summary",
        "model_comparison_full_coefs",
        "model_comparison_interaction",
        "model_comparison_wald_tests",
    }
    assert expected_keys == set(result.keys())
    # Full coefs table should have rows for all models
    coefs = result["model_comparison_full_coefs"]
    assert len(coefs) > 0
    assert set(coefs["model"].unique()) == {"Model 0", "Model 1", "Model 2", "Model 3", "Model 4"}
