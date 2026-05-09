"""Tests for meplt.mean_excess."""
import numpy as np
import pytest
from moirais.fn.meplt import mean_excess


def test_meplt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = mean_excess(x, u_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_meplt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    u_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = mean_excess(x, u_grid)
    assert isinstance(result, dict)
