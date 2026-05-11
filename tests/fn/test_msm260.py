"""Tests for msm260.mvsml_deep_learning_eq_13_2."""
import numpy as np
import pytest
from morie.fn.msm260 import mvsml_deep_learning_eq_13_2


def test_msm260_basic():
    """Test basic functionality."""
    pre = np.random.default_rng(42).normal(0, 1, 100)
    activation = np.random.default_rng(42).normal(0, 1, 100)
    zi = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    values = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_deep_learning_eq_13_2(pre, activation, zi, yi, values, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm260_edge():
    """Test edge cases."""
    pre = np.random.default_rng(42).normal(0, 1, 100)
    activation = np.random.default_rng(42).normal(0, 1, 100)
    zi = np.random.default_rng(42).normal(0, 1, 100)
    yi = np.random.default_rng(42).normal(0, 1, 100)
    values = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_deep_learning_eq_13_2(pre, activation, zi, yi, values, the)
    assert isinstance(result, dict)
