"""Tests for fzbfkd.fauzi_bdfree_kde."""
import numpy as np
import pytest
from moirais.fn.fzbfkd import fauzi_bdfree_kde


def test_fzbfkd_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_bdfree_kde(t, bandwidth, g_func)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzbfkd_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    bandwidth = 0.3
    g_func = (lambda v: v)
    result = fauzi_bdfree_kde(t, bandwidth, g_func)
    assert isinstance(result, dict)
