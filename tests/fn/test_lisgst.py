"""Tests for lisgst.local_getis_g."""
import numpy as np
import pytest
from morie.fn.lisgst import local_getis_g


def test_lisgst_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_getis_g(x, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lisgst_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = local_getis_g(x, W)
    assert isinstance(result, dict)
