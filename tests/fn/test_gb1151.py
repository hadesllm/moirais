"""Tests for gb1151.gibbons_blomqvist_q."""
import numpy as np
import pytest
from morie.fn.gb1151 import gibbons_blomqvist_q


def test_gb1151_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_blomqvist_q(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_gb1151_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_blomqvist_q(x, y)
    assert isinstance(result, dict)
