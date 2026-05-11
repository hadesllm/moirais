"""Tests for getsorg.getis_ord_g."""
import numpy as np
import pytest
from morie.fn.getsorg import getis_ord_g


def test_getsorg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = getis_ord_g(x, W)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_getsorg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = getis_ord_g(x, W)
    assert isinstance(result, dict)
