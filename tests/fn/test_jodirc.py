"""Tests for jodirc.joseph_direct_multistep."""
import numpy as np
import pytest
from morie.fn.jodirc import joseph_direct_multistep


def test_jodirc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_direct_multistep(X, y, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jodirc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_direct_multistep(X, y, H)
    assert isinstance(result, dict)
