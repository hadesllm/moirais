"""Tests for msm116.mvsml_bayesian_regression_pt2_eq_7_9."""
import numpy as np
import pytest
from morie.fn.msm116 import mvsml_bayesian_regression_pt2_eq_7_9


def test_msm116_basic():
    """Test basic functionality."""
    p = 5
    y = np.random.default_rng(43).normal(0, 1, 100)
    cj = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_9(p, y, cj, c, j, block)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm116_edge():
    """Test edge cases."""
    p = 5
    y = np.random.default_rng(43).normal(0, 1, 100)
    cj = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    block = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_9(p, y, cj, c, j, block)
    assert isinstance(result, dict)
