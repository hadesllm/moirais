"""Tests for nmdsf.nonmetric_mds."""
import numpy as np
import pytest
from moirais.fn.nmdsf import nonmetric_mds


def test_nmdsf_basic():
    """Test basic functionality."""
    delta = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = nonmetric_mds(delta, n_dims, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nmdsf_edge():
    """Test edge cases."""
    delta = np.random.default_rng(42).normal(0, 1, 100)
    n_dims = 2
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = nonmetric_mds(delta, n_dims, max_iter)
    assert isinstance(result, dict)
