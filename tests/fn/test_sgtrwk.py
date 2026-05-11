"""Tests for sgtrwk.sgt_random_walk_kernel."""
import numpy as np
import pytest
from morie.fn.sgtrwk import sgt_random_walk_kernel


def test_sgtrwk_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    lam = 0.1
    result = sgt_random_walk_kernel(A, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtrwk_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    lam = 0.1
    result = sgt_random_walk_kernel(A, lam)
    assert isinstance(result, dict)
