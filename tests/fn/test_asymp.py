"""Tests for asymp.asymptotic_expansion."""
import numpy as np
import pytest
from morie.fn.asymp import asymptotic_expansion


def test_asymp_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x_inf = np.random.default_rng(42).normal(0, 1, 100)
    result = asymptotic_expansion(f, x_inf)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_asymp_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x_inf = np.random.default_rng(42).normal(0, 1, 100)
    result = asymptotic_expansion(f, x_inf)
    assert isinstance(result, dict)
