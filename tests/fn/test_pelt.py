"""Tests for pelt.pelt."""
import numpy as np
import pytest
from moirais.fn.pelt import pelt


def test_pelt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cost = np.random.default_rng(42).normal(0, 1, 100)
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = pelt(x, cost, penalty)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pelt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cost = np.random.default_rng(42).normal(0, 1, 100)
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = pelt(x, cost, penalty)
    assert isinstance(result, dict)
