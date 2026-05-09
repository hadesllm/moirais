"""Tests for btdir.boot_dirichlet_weights."""
import numpy as np
import pytest
from moirais.fn.btdir import boot_dirichlet_weights


def test_btdir_basic():
    """Test basic functionality."""
    n = 100
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    rng = np.random.default_rng(42)
    result = boot_dirichlet_weights(n, B, rng)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btdir_edge():
    """Test edge cases."""
    n = 100
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    rng = np.random.default_rng(42)
    result = boot_dirichlet_weights(n, B, rng)
    assert isinstance(result, dict)
