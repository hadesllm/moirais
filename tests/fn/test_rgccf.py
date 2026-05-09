"""Tests for rgccf.rangayyan_ccf."""
import numpy as np
import pytest
from moirais.fn.rgccf import rangayyan_ccf


def test_rgccf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ccf(x, y, max_lag)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rgccf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ccf(x, y, max_lag)
    assert isinstance(result, dict)
