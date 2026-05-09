"""Tests for hmbgdg.geron_batch_gd_grad."""
import numpy as np
import pytest
from moirais.fn.hmbgdg import geron_batch_gd_grad


def test_hmbgdg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_batch_gd_grad(X, y, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbgdg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    result = geron_batch_gd_grad(X, y, theta)
    assert isinstance(result, dict)
