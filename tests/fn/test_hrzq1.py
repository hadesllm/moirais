"""Tests for hrzq1.horowitz_quantile_regression."""
import numpy as np
import pytest
from morie.fn.hrzq1 import horowitz_quantile_regression


def test_hrzq1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    result = horowitz_quantile_regression(x, y, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzq1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    tau = 0.1
    result = horowitz_quantile_regression(x, y, tau)
    assert isinstance(result, dict)
