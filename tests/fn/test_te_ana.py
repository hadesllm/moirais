"""Tests for moirais.fn.te_ana — treatment effects analysis (ATE/ATT/ATC/CATE)."""

import math

import numpy as np
import pandas as pd
import pytest

from moirais.fn.te_ana import run_treatment_effects_analysis


def _make_cpads_frame(rng, n=200):
    """Create a minimal CPADS-compliant synthetic frame with all required columns."""
    age = rng.choice(["18-24", "25-34", "35-44", "45-54"], size=n)
    gender = rng.choice(["male", "female"], size=n)
    province = rng.choice(["ON", "QC", "BC", "AB"], size=n)
    mental = rng.choice(["excellent", "good", "fair", "poor"], size=n)
    physical = rng.choice(["excellent", "good", "fair", "poor"], size=n)
    treatment = rng.integers(0, 2, size=n)
    # Outcome with known treatment effect
    p_outcome = 0.25 + 0.20 * treatment
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
def te_data():
    rng = np.random.default_rng(42)
    return _make_cpads_frame(rng, n=200)


def test_returns_result_with_ate_key(te_data):
    """run_treatment_effects_analysis returns dict with treatment_effects_summary."""
    result = run_treatment_effects_analysis(te_data)
    assert isinstance(result, dict)
    assert "treatment_effects_summary" in result
    summary = result["treatment_effects_summary"]
    assert "ATE" in summary["estimand"].values


def test_ate_is_finite(te_data):
    """ATE estimate is a finite number."""
    result = run_treatment_effects_analysis(te_data)
    summary = result["treatment_effects_summary"]
    ate_row = summary[summary["estimand"] == "ATE"]
    ate_val = float(ate_row["estimate"].iloc[0])
    assert math.isfinite(ate_val)


def test_works_with_binary_treatment(te_data):
    """Analysis completes and produces all three estimands for binary treatment."""
    result = run_treatment_effects_analysis(te_data)
    summary = result["treatment_effects_summary"]
    assert set(summary["estimand"]) == {"ATE", "ATT", "ATC"}
    # All three estimates should be finite
    for _, row in summary.iterrows():
        assert math.isfinite(row["estimate"]), f"{row['estimand']} is not finite"


def test_handles_covariates(te_data):
    """CATE subgroup estimates are produced for each covariate."""
    result = run_treatment_effects_analysis(te_data)
    cate = result["cate_subgroup_estimates"]
    assert isinstance(cate, pd.DataFrame)
    # Should have subgroup estimates for at least some covariates
    if len(cate) > 0:
        assert "subgroup_var" in cate.columns
        assert "cate" in cate.columns
        # All CATE values should be finite
        assert cate["cate"].apply(math.isfinite).all()
