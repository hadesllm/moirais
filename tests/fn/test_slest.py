"""Tests for morie.fn.slest — sieve likelihood estimation."""

import numpy as np
import pytest

from morie.fn.slest import slest


def test_basic_output():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = slest(x)
    assert "density" in result
    assert "coefficients" in result
    assert result["n"] == 200


def test_density_nonnegative():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(300)
    result = slest(x)
    assert np.all(result["density"] >= 0)


def test_density_integrates():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    result = slest(x)
    integral = np.trapezoid(result["density"], result["eval_points"])
    assert integral == pytest.approx(1.0, abs=0.3)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        slest(np.array([]))
