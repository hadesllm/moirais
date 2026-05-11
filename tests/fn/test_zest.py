"""Tests for morie.fn.zest — Z-estimator."""

import numpy as np
import pytest

from morie.fn.zest import zest


def test_mean_estimation():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500) + 3.0
    result = zest(x, psi=lambda xi, theta: xi - theta)
    assert result["theta"] == pytest.approx(3.0, abs=0.3)
    assert result["converged"] is True


def test_ci_contains_true():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000) + 1.0
    result = zest(x, psi=lambda xi, theta: xi - theta)
    assert result["ci_lower"] < 1.0 < result["ci_upper"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        zest(np.array([]), psi=lambda xi, theta: xi - theta)
