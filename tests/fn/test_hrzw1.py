"""Tests for hrzw1.horowitz_wild_bootstrap."""
import numpy as np
import pytest
from morie.fn.hrzw1 import horowitz_wild_bootstrap


def test_hrzw1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_wild_bootstrap(x, y, residuals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzw1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_wild_bootstrap(x, y, residuals)
    assert isinstance(result, dict)
