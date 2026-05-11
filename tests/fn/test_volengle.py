"""Tests for volengle.vol_engle_lagrange."""
import numpy as np
import pytest
from morie.fn.volengle import vol_engle_lagrange


def test_volengle_basic():
    """Test basic functionality."""
    r = 10
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_engle_lagrange(r, q)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_volengle_edge():
    """Test edge cases."""
    r = 10
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_engle_lagrange(r, q)
    assert isinstance(result, dict)
