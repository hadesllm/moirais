"""Tests for erstst.ers_unit_root."""
import numpy as np
import pytest
from morie.fn.erstst import ers_unit_root


def test_erstst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = ers_unit_root(x, lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_erstst_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = ers_unit_root(x, lags)
    assert isinstance(result, dict)
