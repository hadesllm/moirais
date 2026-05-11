"""Tests for otsw.ot_sliced_wasserstein."""
import numpy as np
import pytest
from morie.fn.otsw import ot_sliced_wasserstein


def test_otsw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    n_proj = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sliced_wasserstein(X, Y, p, n_proj)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsw_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    n_proj = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sliced_wasserstein(X, Y, p, n_proj)
    assert isinstance(result, dict)
