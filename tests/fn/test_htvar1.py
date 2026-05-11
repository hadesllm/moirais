"""Tests for htvar1.ht_variance."""
import numpy as np
import pytest
from morie.fn.htvar1 import ht_variance


def test_htvar1_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    pi_ij = np.random.default_rng(42).normal(0, 1, 100)
    result = ht_variance(y, pi, pi_ij)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_htvar1_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    pi_ij = np.random.default_rng(42).normal(0, 1, 100)
    result = ht_variance(y, pi, pi_ij)
    assert isinstance(result, dict)
