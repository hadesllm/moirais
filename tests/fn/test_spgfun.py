"""Tests for spgfun.schabenberger_g_function."""
import numpy as np
import pytest
from moirais.fn.spgfun import schabenberger_g_function


def test_spgfun_basic():
    """Test basic functionality."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_g_function(points, r)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spgfun_edge():
    """Test edge cases."""
    points = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    result = schabenberger_g_function(points, r)
    assert isinstance(result, dict)
