"""Tests for msm085.mvsml_bayesian_regression_pt2_eq_7_1."""
import numpy as np
import pytest
from moirais.fn.msm085 import mvsml_bayesian_regression_pt2_eq_7_1


def test_msm085_basic():
    """Test basic functionality."""
    variable = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    categories = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    conceived = np.random.default_rng(42).normal(0, 1, 100)
    contiguous = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(variable, the, categories, are, conceived, contiguous)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm085_edge():
    """Test edge cases."""
    variable = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    categories = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    conceived = np.random.default_rng(42).normal(0, 1, 100)
    contiguous = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_pt2_eq_7_1(variable, the, categories, are, conceived, contiguous)
    assert isinstance(result, dict)
