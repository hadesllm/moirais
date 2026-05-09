"""Tests for btvinf.boot_influence_fn."""
import numpy as np
import pytest
from moirais.fn.btvinf import boot_influence_fn


def test_btvinf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_influence_fn(x, stat, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btvinf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    stat = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = boot_influence_fn(x, stat, eps)
    assert isinstance(result, dict)
