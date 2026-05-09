"""Tests for fzbfkde2.fauzi_bdfree_density_from_cdf."""
import numpy as np
import pytest
from moirais.fn.fzbfkde2 import fauzi_bdfree_density_from_cdf


def test_fzbfkde2_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_bdfree_density_from_cdf(x, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzbfkde2_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_bdfree_density_from_cdf(x, bandwidth, g_func)
    assert isinstance(result, dict)
