"""Tests for jolagf.joseph_lag_feature."""
import numpy as np
import pytest
from morie.fn.jolagf import joseph_lag_feature


def test_jolagf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = joseph_lag_feature(y, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jolagf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    result = joseph_lag_feature(y, k)
    assert isinstance(result, dict)
