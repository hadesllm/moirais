"""Tests for morie.fn.plmod — Partially linear model (Robinson 1988)."""

import numpy as np
import pytest

from morie.fn.plmod import plmod


@pytest.fixture()
def plm_data():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.uniform(0, 2 * np.pi, n)
    d = 0.5 * np.sin(x) + rng.normal(0, 0.3, n)
    y = 2.0 * d + np.sin(x) + rng.normal(0, 0.3, n)
    return y, d, x


def test_returns_dict(plm_data):
    y, d, x = plm_data
    result = plmod(y, d, x)
    assert isinstance(result, dict)
    for key in ("theta", "se", "t_stat", "pval", "ci_lower", "ci_upper", "bandwidth", "n_obs"):
        assert key in result


def test_theta_near_true(plm_data):
    y, d, x = plm_data
    result = plmod(y, d, x)
    assert abs(result["theta"] - 2.0) < 1.0


def test_se_positive(plm_data):
    y, d, x = plm_data
    result = plmod(y, d, x)
    assert result["se"] > 0


def test_ci_contains_theta(plm_data):
    y, d, x = plm_data
    result = plmod(y, d, x)
    assert result["ci_lower"] <= result["theta"] <= result["ci_upper"]


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="same number"):
        plmod(np.ones(10), np.ones(5), np.ones(10))


def test_too_few():
    with pytest.raises(ValueError, match="at least 10"):
        plmod(np.ones(5), np.ones(5), np.ones(5))


def test_custom_bandwidth(plm_data):
    y, d, x = plm_data
    result = plmod(y, d, x, bandwidth=0.5)
    assert np.isfinite(result["theta"])
