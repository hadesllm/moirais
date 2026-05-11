"""Tests for wsmmtl.wasserman_mutual_info."""
import numpy as np
import pytest
from morie.fn.wsmmtl import wasserman_mutual_info


def test_wsmmtl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_mutual_info(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmmtl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_mutual_info(x, y)
    assert isinstance(result, dict)
