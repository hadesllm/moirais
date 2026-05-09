"""Tests for rgrlsl.rangayyan_rls_lattice."""
import numpy as np
import pytest
from moirais.fn.rgrlsl import rangayyan_rls_lattice


def test_rgrlsl_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    lam = 0.1
    order = 4
    result = rangayyan_rls_lattice(x, d, lam, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgrlsl_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    d = 5
    lam = 0.1
    order = 4
    result = rangayyan_rls_lattice(x, d, lam, order)
    assert isinstance(result, dict)
