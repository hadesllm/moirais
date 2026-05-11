"""Tests for morie.fn.g_comp — G-computation ATE estimator with bootstrap SE."""

import math

import numpy as np
import pandas as pd
import pytest

from morie.fn.g_comp import estimate_ate_gcomputation


@pytest.fixture()
def gcomp_data():
    """Synthetic data with binary treatment and continuous outcome."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.0 + 1.5 * t + 0.8 * x1 - 0.3 * x2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_returns_dict_with_keys(gcomp_data):
    """estimate_ate_gcomputation returns dict with required keys."""
    result = estimate_ate_gcomputation(
        gcomp_data, treatment="t", outcome="y", covariates=["x1", "x2"],
    )
    assert isinstance(result, dict)
    for key in ("ate", "se", "ci_lower", "ci_upper", "n_obs", "outcome_model"):
        assert key in result


def test_ci_contains_ate(gcomp_data):
    """Bootstrap CI should contain the point estimate."""
    result = estimate_ate_gcomputation(
        gcomp_data, treatment="t", outcome="y", covariates=["x1", "x2"],
    )
    assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]


def test_bootstrap_se_finite(gcomp_data):
    """Bootstrap SE should be finite and positive."""
    result = estimate_ate_gcomputation(
        gcomp_data, treatment="t", outcome="y", covariates=["x1", "x2"],
    )
    assert math.isfinite(result["se"])
    assert result["se"] > 0


def test_invalid_outcome_model_raises(gcomp_data):
    """Invalid outcome_model should raise ValueError."""
    with pytest.raises(ValueError, match="outcome_model"):
        estimate_ate_gcomputation(
            gcomp_data, treatment="t", outcome="y",
            covariates=["x1", "x2"], outcome_model="invalid",
        )


def test_too_few_observations_raises():
    """Fewer than 10 rows should raise ValueError."""
    df = pd.DataFrame({
        "y": [1, 2, 3], "t": [0, 1, 0], "x1": [0.1, 0.2, 0.3],
    })
    with pytest.raises(ValueError, match="at least 10"):
        estimate_ate_gcomputation(
            df, treatment="t", outcome="y", covariates=["x1"],
        )
