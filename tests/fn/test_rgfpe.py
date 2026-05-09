"""Tests for rgfpe.rangayyan_ar_order_fpe."""
import numpy as np
import pytest
from moirais.fn.rgfpe import rangayyan_ar_order_fpe


def test_rgfpe_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_order_fpe(x, max_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgfpe_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_order_fpe(x, max_order)
    assert isinstance(result, dict)
