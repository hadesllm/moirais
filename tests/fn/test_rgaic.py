"""Tests for rgaic.rangayyan_ar_order_aic."""
import numpy as np
import pytest
from moirais.fn.rgaic import rangayyan_ar_order_aic


def test_rgaic_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_order_aic(x, max_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgaic_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_order_aic(x, max_order)
    assert isinstance(result, dict)
