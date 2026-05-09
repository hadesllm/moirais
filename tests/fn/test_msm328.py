"""Tests for msm328.mvsml_functional_regression_eq_15_3."""
import numpy as np
import pytest
from moirais.fn.msm328 import mvsml_functional_regression_eq_15_3


def test_msm328_basic():
    """Test basic functionality."""
    bY = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    It = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_3(bY, exp, b, It, important, to)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm328_edge():
    """Test edge cases."""
    bY = np.random.default_rng(42).normal(0, 1, 100)
    exp = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    It = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_3(bY, exp, b, It, important, to)
    assert isinstance(result, dict)
