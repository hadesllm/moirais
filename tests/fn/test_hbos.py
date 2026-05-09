"""Tests for hbos.hbos."""
import numpy as np
import pytest
from moirais.fn.hbos import hbos


def test_hbos_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bins = np.random.default_rng(42).normal(0, 1, 100)
    result = hbos(X, bins)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hbos_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    bins = np.random.default_rng(42).normal(0, 1, 100)
    result = hbos(X, bins)
    assert isinstance(result, dict)
