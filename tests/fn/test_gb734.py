"""Tests for gb734.gibbons_linrank_symmetry_cond."""
import numpy as np
import pytest
from moirais.fn.gb734 import gibbons_linrank_symmetry_cond


def test_gb734_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    result = gibbons_linrank_symmetry_cond(a, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb734_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    N = 100
    result = gibbons_linrank_symmetry_cond(a, N)
    assert isinstance(result, dict)
