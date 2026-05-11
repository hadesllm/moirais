"""Tests for morie.fn.empll — empirical likelihood ratio."""

import numpy as np
import pytest

from morie.fn.empll import empll


def test_true_mean_not_rejected():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = empll(x, mu0=0.0)
    assert result["reject"] is False
    assert result["p_value"] > 0.05


def test_false_mean_rejected():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500) + 3.0
    result = empll(x, mu0=0.0)
    assert result["reject"] is True


def test_weights_sum_to_one():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(100)
    result = empll(x, mu0=0.0)
    assert np.sum(result["weights"]) == pytest.approx(1.0, abs=1e-6)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        empll(np.array([]))
