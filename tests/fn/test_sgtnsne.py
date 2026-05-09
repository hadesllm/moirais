"""Tests for sgtnsne.sgt_isomap."""
import numpy as np
import pytest
from moirais.fn.sgtnsne import sgt_isomap


def test_sgtnsne_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_isomap(X, k, dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtnsne_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    k = 5
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = sgt_isomap(X, k, dim)
    assert isinstance(result, dict)
