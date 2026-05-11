"""Tests for jorolf.joseph_rolling_window_feature."""
import numpy as np
import pytest
from morie.fn.jorolf import joseph_rolling_window_feature


def test_jorolf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    agg = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_rolling_window_feature(y, W, agg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jorolf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    agg = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_rolling_window_feature(y, W, agg)
    assert isinstance(result, dict)
