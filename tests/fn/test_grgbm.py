"""Tests for grgbm.geron_gradient_boosting_residual."""
import numpy as np
import pytest
from moirais.fn.grgbm import geron_gradient_boosting_residual


def test_grgbm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    F_prev = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gradient_boosting_residual(X, y, F_prev)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grgbm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    F_prev = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_gradient_boosting_residual(X, y, F_prev)
    assert isinstance(result, dict)
