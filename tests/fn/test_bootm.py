"""Tests for morie.fn.bootm — Bootstrap for M-estimators."""

import numpy as np
import pytest

from morie.fn.bootm import bootm


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


def test_returns_dict(data):
    result = bootm(data)
    assert isinstance(result, dict)
    for key in ("estimate", "se", "ci_lower", "ci_upper", "boot_estimates", "n_boot", "method"):
        assert key in result


def test_estimate_is_mean(data):
    result = bootm(data)
    np.testing.assert_allclose(result["estimate"], np.mean(data))


def test_se_positive(data):
    result = bootm(data)
    assert result["se"] > 0


def test_ci_brackets_estimate(data):
    result = bootm(data)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_boot_estimates_length(data):
    result = bootm(data, n_boot=500)
    assert len(result["boot_estimates"]) == 500


def test_multiplier_bootstrap(data):
    result = bootm(data, method="multiplier")
    assert "multiplier" in result["method"]
    assert np.isfinite(result["se"])


def test_custom_estimator(data):
    result = bootm(data, estimator=np.median)
    assert np.isfinite(result["estimate"])
    np.testing.assert_allclose(result["estimate"], np.median(data))


def test_deterministic_with_seed(data):
    r1 = bootm(data, seed=7)
    r2 = bootm(data, seed=7)
    np.testing.assert_array_equal(r1["boot_estimates"], r2["boot_estimates"])


def test_2d_data():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 3))
    result = bootm(data)
    assert np.isfinite(result["estimate"])
