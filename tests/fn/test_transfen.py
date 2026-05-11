"""Tests for transfen.transfer_entropy_te."""
import numpy as np
import pytest
from morie.fn.transfen import transfer_entropy_te


def test_transfen_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag = 10
    result = transfer_entropy_te(x, y, lag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_transfen_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag = 10
    result = transfer_entropy_te(x, y, lag)
    assert isinstance(result, dict)
