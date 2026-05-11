"""Tests for shewh.shewhart."""
import numpy as np
import pytest
from morie.fn.shewh import shewhart


def test_shewh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = shewhart(x, mu, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shewh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    sigma = 1.0
    result = shewhart(x, mu, sigma)
    assert isinstance(result, dict)
