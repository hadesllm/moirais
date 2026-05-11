"""Tests for irt3pl.three_parameter_logistic."""
import numpy as np
import pytest
from morie.fn.irt3pl import three_parameter_logistic


def test_irt3pl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = three_parameter_logistic(y, theta, a, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irt3pl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = three_parameter_logistic(y, theta, a, b, c)
    assert isinstance(result, dict)
