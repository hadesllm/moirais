"""Tests for bkfour.baron_kenny_four_step."""
import numpy as np
import pytest
from morie.fn.bkfour import baron_kenny_four_step


def test_bkfour_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = baron_kenny_four_step(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkfour_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = baron_kenny_four_step(X, M, Y)
    assert isinstance(result, dict)
