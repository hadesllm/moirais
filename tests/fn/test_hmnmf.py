"""Tests for hmnmf.geron_nmf."""
import numpy as np
import pytest
from moirais.fn.hmnmf import geron_nmf


def test_hmnmf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_nmf(X, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmnmf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    result = geron_nmf(X, n_components)
    assert isinstance(result, dict)
