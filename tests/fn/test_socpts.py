"""Tests for socpts.second_order_cone."""
import numpy as np
import pytest
from moirais.fn.socpts import second_order_cone


def test_socpts_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    domains = np.random.default_rng(42).normal(0, 1, 100)
    result = second_order_cone(c, A, b, domains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_socpts_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    domains = np.random.default_rng(42).normal(0, 1, 100)
    result = second_order_cone(c, A, b, domains)
    assert isinstance(result, dict)
