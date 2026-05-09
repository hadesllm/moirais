"""Tests for msm105.mvsml_bayesian_regression_pt2_eq_7_2."""
import numpy as np
import pytest
from moirais.fn.msm105 import mvsml_bayesian_regression_pt2_eq_7_2


def test_msm105_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    scaled = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    chi = np.random.default_rng(42).normal(0, 1, 100)
    squared = np.random.default_rng(42).normal(0, 1, 100)
    distribution = 'normal'
    result = mvsml_bayesian_regression_pt2_eq_7_2(a, scaled, inverse, chi, squared, distribution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm105_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    scaled = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    chi = np.random.default_rng(42).normal(0, 1, 100)
    squared = np.random.default_rng(42).normal(0, 1, 100)
    distribution = 'normal'
    result = mvsml_bayesian_regression_pt2_eq_7_2(a, scaled, inverse, chi, squared, distribution)
    assert isinstance(result, dict)
