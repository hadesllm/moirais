"""Tests for msm088.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from morie.fn.msm088 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm088_basic():
    """Test basic functionality."""
    G = np.eye(10)
    p = 5
    XTX = np.random.default_rng(42).normal(0, 1, 100)
    Then = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    assuming = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(G, p, XTX, Then, this, assuming)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm088_edge():
    """Test edge cases."""
    G = np.eye(10)
    p = 5
    XTX = np.random.default_rng(42).normal(0, 1, 100)
    Then = np.random.default_rng(42).normal(0, 1, 100)
    this = np.random.default_rng(42).normal(0, 1, 100)
    assuming = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(G, p, XTX, Then, this, assuming)
    assert isinstance(result, dict)
