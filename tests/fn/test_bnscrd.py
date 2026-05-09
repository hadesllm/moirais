"""Tests for bnscrd.bound_causal_rd."""
import numpy as np
import pytest
from moirais.fn.bnscrd import bound_causal_rd


def test_bnscrd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = bound_causal_rd(y, x, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnscrd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    result = bound_causal_rd(y, x, cutoff)
    assert isinstance(result, dict)
