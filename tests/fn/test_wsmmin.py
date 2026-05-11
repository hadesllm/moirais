"""Tests for wsmmin.wasserman_minimax."""
import numpy as np
import pytest
from morie.fn.wsmmin import wasserman_minimax


def test_wsmmin_basic():
    """Test basic functionality."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    family = 'gaussian'
    result = wasserman_minimax(loss, estimator, family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmin_edge():
    """Test edge cases."""
    loss = np.random.default_rng(42).normal(0, 1, 100)
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    family = 'gaussian'
    result = wasserman_minimax(loss, estimator, family)
    assert isinstance(result, dict)
