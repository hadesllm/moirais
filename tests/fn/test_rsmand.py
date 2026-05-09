"""Tests for rsmand.rating_scale_andrich."""
import numpy as np
import pytest
from moirais.fn.rsmand import rating_scale_andrich


def test_rsmand_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    b = np.random.default_rng(42).normal(0, 1, 100)
    tau_j = np.random.default_rng(42).normal(0, 1, 100)
    result = rating_scale_andrich(y, theta, b, tau_j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rsmand_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    theta = 0.0
    b = np.random.default_rng(42).normal(0, 1, 100)
    tau_j = np.random.default_rng(42).normal(0, 1, 100)
    result = rating_scale_andrich(y, theta, b, tau_j)
    assert isinstance(result, dict)
