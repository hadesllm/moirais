"""Tests for dpkb.k_anonymity."""
import numpy as np
import pytest
from moirais.fn.dpkb import k_anonymity


def test_dpkb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quasi_ids = np.arange(100, dtype=int)
    k = 5
    result = k_anonymity(X, quasi_ids, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpkb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quasi_ids = np.arange(100, dtype=int)
    k = 5
    result = k_anonymity(X, quasi_ids, k)
    assert isinstance(result, dict)
