"""Tests for msm329.mvsml_functional_regression_eq_15_4."""
import numpy as np
import pytest
from moirais.fn.msm329 import mvsml_functional_regression_eq_15_4


def test_msm329_basic():
    """Test basic functionality."""
    It = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_4(It, important, to, point, out, that)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm329_edge():
    """Test edge cases."""
    It = np.random.default_rng(42).normal(0, 1, 100)
    important = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    point = np.random.default_rng(42).normal(0, 1, 100)
    out = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_4(It, important, to, point, out, that)
    assert isinstance(result, dict)
