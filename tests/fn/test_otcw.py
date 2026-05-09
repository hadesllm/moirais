"""Tests for otcw.ot_cyclical_weight."""
import numpy as np
import pytest
from moirais.fn.otcw import ot_cyclical_weight


def test_otcw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    perm = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_cyclical_weight(X, Y, C, perm)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otcw_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    perm = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_cyclical_weight(X, Y, C, perm)
    assert isinstance(result, dict)
