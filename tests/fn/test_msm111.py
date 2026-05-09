"""Tests for msm111.mvsml_bayesian_regression_pt2_eq_7_7."""
import numpy as np
import pytest
from moirais.fn.msm111 import mvsml_bayesian_regression_pt2_eq_7_7


def test_msm111_basic():
    """Test basic functionality."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    When = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    large = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    direct = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_7(l, When, p, large, n, direct)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm111_edge():
    """Test edge cases."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    When = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    large = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    direct = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_7(l, When, p, large, n, direct)
    assert isinstance(result, dict)
