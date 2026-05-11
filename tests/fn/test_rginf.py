"""Tests for rginf.rangayyan_infomax_ica."""
import numpy as np
import pytest
from morie.fn.rginf import rangayyan_infomax_ica


def test_rginf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    lr = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_infomax_ica(X, n_components, lr, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rginf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_components = 3
    lr = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_infomax_ica(X, n_components, lr, max_iter)
    assert isinstance(result, dict)
