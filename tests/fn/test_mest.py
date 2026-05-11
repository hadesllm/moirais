"""Tests for morie.fn.mest — M-estimator."""

import numpy as np
import pytest

from morie.fn.mest import mest


def test_mean_via_ls():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500) + 2.0
    result = mest(x, rho=lambda xi, theta: (xi - theta) ** 2)
    assert result["theta"] == pytest.approx(2.0, abs=0.3)


def test_median_via_abs():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    result = mest(x, rho=lambda xi, theta: abs(xi - theta))
    assert result["theta"] == pytest.approx(np.median(x), abs=0.2)


def test_converged():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = mest(x, rho=lambda xi, theta: (xi - theta) ** 2)
    assert result["converged"] is True


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mest(np.array([]), rho=lambda xi, theta: (xi - theta) ** 2)
