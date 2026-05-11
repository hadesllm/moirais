"""Tests for neymal.neyman_allocation."""
import numpy as np
import pytest
from morie.fn.neymal import neyman_allocation


def test_neymal_basic():
    """Test basic functionality."""
    N = 100
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    Sh = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = neyman_allocation(N, Nh, Sh, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_neymal_edge():
    """Test edge cases."""
    N = 100
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    Sh = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = neyman_allocation(N, Nh, Sh, n)
    assert isinstance(result, dict)
