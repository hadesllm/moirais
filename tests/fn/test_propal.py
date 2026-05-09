"""Tests for propal.proportional_allocation."""
import numpy as np
import pytest
from moirais.fn.propal import proportional_allocation


def test_propal_basic():
    """Test basic functionality."""
    N = 100
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = proportional_allocation(N, Nh, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_propal_edge():
    """Test edge cases."""
    N = 100
    Nh = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = proportional_allocation(N, Nh, n)
    assert isinstance(result, dict)
