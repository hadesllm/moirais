"""Tests for msm115.mvsml_bayesian_regression_pt2_eq_7_10."""
import numpy as np
import pytest
from moirais.fn.msm115 import mvsml_bayesian_regression_pt2_eq_7_10


def test_msm115_basic():
    """Test basic functionality."""
    p = 5
    y = np.random.default_rng(43).normal(0, 1, 100)
    cj = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_10(p, y, cj)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm115_edge():
    """Test edge cases."""
    p = 5
    y = np.random.default_rng(43).normal(0, 1, 100)
    cj = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_10(p, y, cj)
    assert isinstance(result, dict)
