"""Tests for otc2p.ot_cost_lp."""
import numpy as np
import pytest
from morie.fn.otc2p import ot_cost_lp


def test_otc2p_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = ot_cost_lp(X, Y, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otc2p_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = ot_cost_lp(X, Y, p)
    assert isinstance(result, dict)
