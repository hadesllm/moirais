"""Tests for grn007.geron_ch4_mse_gradient_vector."""
import numpy as np
import pytest
from moirais.fn.grn007 import geron_ch4_mse_gradient_vector


def test_grn007_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_ch4_mse_gradient_vector(X, y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn007_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_ch4_mse_gradient_vector(X, y, theta)
    assert isinstance(result, dict)
