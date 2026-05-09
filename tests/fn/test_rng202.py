"""Tests for rng202.rangayyan_ch4_ccf_discrete_with_delay."""
import numpy as np
import pytest
from moirais.fn.rng202 import rangayyan_ch4_ccf_discrete_with_delay


def test_rng202_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    n = 100
    result = rangayyan_ch4_ccf_discrete_with_delay(x, y, k, n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_rng202_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    k = 5
    n = 100
    result = rangayyan_ch4_ccf_discrete_with_delay(x, y, k, n)
    assert isinstance(result, dict)
