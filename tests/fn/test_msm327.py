"""Tests for msm327.mvsml_functional_regression_eq_15_3."""
import numpy as np
import pytest
from morie.fn.msm327 import mvsml_functional_regression_eq_15_3


def test_msm327_basic():
    """Test basic functionality."""
    LL = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    node = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_3(LL, right, node, are, the, log)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm327_edge():
    """Test edge cases."""
    LL = np.random.default_rng(42).normal(0, 1, 100)
    right = np.random.default_rng(42).normal(0, 1, 100)
    node = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    log = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_3(LL, right, node, are, the, log)
    assert isinstance(result, dict)
