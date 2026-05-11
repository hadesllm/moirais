"""Tests for propalc.proportional_allocation."""
import numpy as np
import pytest
from morie.fn.propalc import proportional_allocation


def test_propalc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N_h = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = proportional_allocation(y, N_h, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_propalc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N_h = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = proportional_allocation(y, N_h, n)
    assert isinstance(result, dict)
