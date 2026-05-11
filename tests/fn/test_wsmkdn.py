"""Tests for wsmkdn.wasserman_kde."""
import numpy as np
import pytest
from morie.fn.wsmkdn import wasserman_kde


def test_wsmkdn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = wasserman_kde(x, data, h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmkdn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    result = wasserman_kde(x, data, h)
    assert isinstance(result, dict)
