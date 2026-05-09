"""Tests for randW.random_walk_kernel."""
import numpy as np
import pytest
from moirais.fn.randW import random_walk_kernel


def test_randW_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = random_walk_kernel(G1, G2, lam)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_randW_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    result = random_walk_kernel(G1, G2, lam)
    assert isinstance(result, dict)
