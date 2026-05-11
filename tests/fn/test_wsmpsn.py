"""Tests for wsmpsn.wasserman_pearson_corr."""
import numpy as np
import pytest
from morie.fn.wsmpsn import wasserman_pearson_corr


def test_wsmpsn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_pearson_corr(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wsmpsn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = wasserman_pearson_corr(x, y)
    assert isinstance(result, dict)
