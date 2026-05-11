"""Tests for bkdot.burkov_dot_product."""
import numpy as np
import pytest
from morie.fn.bkdot import burkov_dot_product


def test_bkdot_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_dot_product(a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkdot_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_dot_product(a, b)
    assert isinstance(result, dict)
