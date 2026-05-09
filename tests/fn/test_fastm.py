"""Tests for fastm.fast_mcd."""
import numpy as np
import pytest
from moirais.fn.fastm import fast_mcd


def test_fastm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    n_starts = np.random.default_rng(42).normal(0, 1, 100)
    result = fast_mcd(X, h, n_starts)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fastm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    h = 0.3
    n_starts = np.random.default_rng(42).normal(0, 1, 100)
    result = fast_mcd(X, h, n_starts)
    assert isinstance(result, dict)
