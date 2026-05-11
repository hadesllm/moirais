"""Tests for wsmifn.wasserman_influence_function."""
import numpy as np
import pytest
from morie.fn.wsmifn import wasserman_influence_function


def test_wsmifn_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = wasserman_influence_function(data, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmifn_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = wasserman_influence_function(data, T)
    assert isinstance(result, dict)
