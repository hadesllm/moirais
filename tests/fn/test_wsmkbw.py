"""Tests for wsmkbw.wasserman_kde_bandwidth."""
import numpy as np
import pytest
from moirais.fn.wsmkbw import wasserman_kde_bandwidth


def test_wsmkbw_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_kde_bandwidth(data)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmkbw_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_kde_bandwidth(data)
    assert isinstance(result, dict)
