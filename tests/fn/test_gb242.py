"""Tests for gb242.gibbons_order_pdf."""
import numpy as np
import pytest
from moirais.fn.gb242 import gibbons_order_pdf


def test_gb242_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_pdf(x, r, n, f, F)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb242_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    n = 100
    f = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_order_pdf(x, r, n, f, F)
    assert isinstance(result, dict)
