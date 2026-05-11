"""Tests for morie.fn.wlog — weighted logistic analysis with interaction and SMOTE."""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from morie.fn.wlog import run_weighted_logistic_analysis


def _make_cpads_frame(rng, n=200):
    """Create a minimal CPADS-compliant synthetic frame with all required columns."""
    age = rng.choice(["18-24", "25-34", "35-44", "45-54"], size=n)
    gender = rng.choice(["male", "female"], size=n)
    province = rng.choice(["ON", "QC", "BC", "AB"], size=n)
    mental = rng.choice(["excellent", "good", "fair", "poor"], size=n)
    physical = rng.choice(["excellent", "good", "fair", "poor"], size=n)
    treatment = rng.integers(0, 2, size=n)
    # Outcome correlated with treatment
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
def wlog_data():
    rng = np.random.default_rng(42)
    return _make_cpads_frame(rng, n=200)


def test_returns_dict(wlog_data):
    """run_weighted_logistic_analysis returns a dict with expected keys."""
    result = run_weighted_logistic_analysis(wlog_data)
    assert isinstance(result, dict)
    assert "logistic_odds_ratios" in result
    assert "logistic_interaction_tests" in result


def test_or_table_is_dataframe(wlog_data):
    """The odds ratio table is a DataFrame with expected columns."""
    result = run_weighted_logistic_analysis(wlog_data)
    or_table = result["logistic_odds_ratios"]
    assert isinstance(or_table, pd.DataFrame)
    assert "term" in or_table.columns
    assert "OR" in or_table.columns


def test_coefficients_are_finite(wlog_data):
    """All odds ratios and log-odds in the main model are finite."""
    result = run_weighted_logistic_analysis(wlog_data)
    or_table = result["logistic_odds_ratios"]
    assert or_table["log_odds"].apply(math.isfinite).all()
    assert or_table["OR"].apply(math.isfinite).all()


def test_handles_weighted_data(wlog_data):
    """Analysis runs without error when weights vary across observations."""
    # Modify weights to be more extreme
    wlog_data["weight"] = np.where(
        wlog_data["cannabis_any_use"] == 1, 2.5, 0.8
    )
    result = run_weighted_logistic_analysis(wlog_data)
    assert len(result["analysis_frame"]) > 0
    assert result["logistic_odds_ratios"].shape[0] > 0
