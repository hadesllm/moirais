"""Tests for morie.fn.sub_est — Design-based subpopulation mean estimator."""

import math

import numpy as np
import pandas as pd
import pytest

from morie.fn.sub_est import subpopulation_estimate


@pytest.fixture()
def sub_data():
    """Full sample with a domain column and survey weights."""
    rng = np.random.default_rng(42)
    n = 200
    domain = rng.choice(["urban", "rural"], size=n, p=[0.6, 0.4])
    y = np.where(domain == "urban", rng.normal(50, 10, n), rng.normal(30, 8, n))
    w = rng.uniform(1, 5, size=n)
    return pd.DataFrame({"domain": domain, "y": y, "w": w})


def test_returns_dict_with_keys(sub_data):
    """subpopulation_estimate returns dict with required keys."""
    result = subpopulation_estimate(
        sub_data, domain_col="domain", domain_value="urban",
        outcome_col="y", weight_col="w",
    )
    assert isinstance(result, dict)
    for key in ("mean", "se", "ci_lower", "ci_upper", "n_domain"):
        assert key in result


def test_domain_mean_reasonable(sub_data):
    """Urban mean should be higher than rural mean (by data construction)."""
    urban = subpopulation_estimate(
        sub_data, "domain", "urban", "y", "w",
    )
    rural = subpopulation_estimate(
        sub_data, "domain", "rural", "y", "w",
    )
    assert urban["mean"] > rural["mean"]


def test_se_finite_and_positive(sub_data):
    """SE should be finite and positive."""
    result = subpopulation_estimate(
        sub_data, "domain", "urban", "y", "w",
    )
    assert math.isfinite(result["se"])
    assert result["se"] > 0


def test_no_matching_domain_raises(sub_data):
    """Non-existent domain value should raise ValueError."""
    with pytest.raises(ValueError, match="No observations"):
        subpopulation_estimate(
            sub_data, "domain", "suburban", "y", "w",
        )


def test_missing_column_raises(sub_data):
    """Missing column should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        subpopulation_estimate(
            sub_data, "nonexistent", "urban", "y", "w",
        )
