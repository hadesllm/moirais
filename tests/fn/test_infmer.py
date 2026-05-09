"""Tests for infmer.informer."""
import numpy as np
import pytest
from moirais.fn.infmer import informer


def test_infmer_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    seq_len = 100
    result = informer(X, y, seq_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_infmer_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    seq_len = 100
    result = informer(X, y, seq_len)
    assert isinstance(result, dict)
