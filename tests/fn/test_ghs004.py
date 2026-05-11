"""Tests for ghs004.ghosal_ch2_exponential_link_density."""
import numpy as np
import pytest
from morie.fn.ghs004 import ghosal_ch2_exponential_link_density


def test_ghs004_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    result = ghosal_ch2_exponential_link_density(f, x, mu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs004_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    result = ghosal_ch2_exponential_link_density(f, x, mu)
    assert isinstance(result, dict)
