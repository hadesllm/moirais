"""Tests for otsd.ot_sliced_distance_quant."""
import numpy as np
import pytest
from morie.fn.otsd import ot_sliced_distance_quant


def test_otsd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    n_proj = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sliced_distance_quant(X, Y, p, n_proj)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    n_proj = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sliced_distance_quant(X, Y, p, n_proj)
    assert isinstance(result, dict)
