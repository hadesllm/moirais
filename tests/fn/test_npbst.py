"""Tests for morie.fn.npbst -- Nonparametric bootstrap inference."""

import numpy as np
import pytest

from morie.fn.npbst import npbst


@pytest.fixture()
def sample_data():
    rng = np.random.default_rng(99)
    return rng.standard_normal(100)


def test_returns_dict(sample_data):
    result = npbst(sample_data, np.mean)
    assert isinstance(result, dict)
    for k in ("estimate", "se", "ci_lower", "ci_upper", "n_boot", "method"):
        assert k in result


def test_estimate_matches_statistic(sample_data):
    result = npbst(sample_data, np.mean)
    assert result["estimate"] == pytest.approx(np.mean(sample_data))


def test_ci_contains_estimate(sample_data):
    result = npbst(sample_data, np.mean)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


def test_se_positive(sample_data):
    result = npbst(sample_data, np.mean)
    assert result["se"] > 0


def test_basic_method(sample_data):
    result = npbst(sample_data, np.mean, method="basic")
    assert result["method"] == "basic"
    assert result["ci_lower"] <= result["ci_upper"]


def test_normal_method(sample_data):
    result = npbst(sample_data, np.mean, method="normal")
    assert result["method"] == "normal"
    assert np.isfinite(result["ci_lower"])


def test_invalid_method(sample_data):
    with pytest.raises(ValueError, match="method"):
        npbst(sample_data, np.mean, method="bad")


def test_invalid_alpha(sample_data):
    with pytest.raises(ValueError, match="alpha"):
        npbst(sample_data, np.mean, alpha=0)


def test_reproducible(sample_data):
    r1 = npbst(sample_data, np.mean, seed=123)
    r2 = npbst(sample_data, np.mean, seed=123)
    assert r1["se"] == r2["se"]


def test_cheatsheet():
    from morie.fn.npbst import cheatsheet
    assert "bootstrap" in cheatsheet().lower()
