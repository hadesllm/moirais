"""Tests for msm087.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from morie.fn.msm087 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm087_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    scaled = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    chi = np.random.default_rng(42).normal(0, 1, 100)
    squared = np.random.default_rng(42).normal(0, 1, 100)
    From = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(a, scaled, inverse, chi, squared, From)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm087_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    scaled = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    chi = np.random.default_rng(42).normal(0, 1, 100)
    squared = np.random.default_rng(42).normal(0, 1, 100)
    From = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(a, scaled, inverse, chi, squared, From)
    assert isinstance(result, dict)
