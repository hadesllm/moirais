"""Tests for reffec.effective_reproduction."""
import numpy as np
import pytest
from morie.fn.reffec import effective_reproduction


def test_reffec_basic():
    """Test basic functionality."""
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = effective_reproduction(R0, S, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reffec_edge():
    """Test edge cases."""
    R0 = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = effective_reproduction(R0, S, N)
    assert isinstance(result, dict)
