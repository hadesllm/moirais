"""Tests for otmarsh.ot_marginal_shift."""
import numpy as np
import pytest
from morie.fn.otmarsh import ot_marginal_shift


def test_otmarsh_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_marginal_shift(a, b, C, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmarsh_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_marginal_shift(a, b, C, delta)
    assert isinstance(result, dict)
