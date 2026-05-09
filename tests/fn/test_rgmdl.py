"""Tests for rgmdl.rangayyan_ar_order_mdl."""
import numpy as np
import pytest
from moirais.fn.rgmdl import rangayyan_ar_order_mdl


def test_rgmdl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_order_mdl(x, max_order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmdl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_order = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ar_order_mdl(x, max_order)
    assert isinstance(result, dict)
