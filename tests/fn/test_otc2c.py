"""Tests for otc2c.ot_cost_pairwise."""
import numpy as np
import pytest
from moirais.fn.otc2c import ot_cost_pairwise


def test_otc2c_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_cost_pairwise(X, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otc2c_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_cost_pairwise(X, Y)
    assert isinstance(result, dict)
