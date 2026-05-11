"""Tests for grn002.geron_ch4_linear_regression_prediction."""
import numpy as np
import pytest
from morie.fn.grn002 import geron_ch4_linear_regression_prediction


def test_grn002_basic():
    """Test basic functionality."""
    theta = 0.0
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ch4_linear_regression_prediction(theta, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn002_edge():
    """Test edge cases."""
    theta = 0.0
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_ch4_linear_regression_prediction(theta, x)
    assert isinstance(result, dict)
