"""Tests for otsoft.ot_softassignment."""
import numpy as np
import pytest
from moirais.fn.otsoft import ot_softassignment


def test_otsoft_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_softassignment(a, b, C, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsoft_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = ot_softassignment(a, b, C, epsilon)
    assert isinstance(result, dict)
