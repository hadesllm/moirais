"""Tests for morie.fn.sceff — score-based efficient estimation."""

import numpy as np
import pytest

from morie.fn.sceff import sceff


def test_normal_mean():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500) + 3.0
    result = sceff(x, score_func=lambda xi, theta: xi - theta)
    assert result["theta"] == pytest.approx(3.0, abs=0.3)
    assert result["converged"] is True


def test_se_positive():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    result = sceff(x, score_func=lambda xi, theta: xi - theta)
    assert result["se"] > 0


def test_ci_contains_true():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000) + 1.0
    result = sceff(x, score_func=lambda xi, theta: xi - theta)
    assert result["ci_lower"] < 1.0 < result["ci_upper"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sceff(np.array([]), score_func=lambda xi, theta: xi - theta)
