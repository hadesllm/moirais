"""Tests for joitran.joseph_itransformer."""
import numpy as np
import pytest
from morie.fn.joitran import joseph_itransformer


def test_joitran_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_variates = np.random.default_rng(42).normal(0, 1, 100)
    transformer = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_itransformer(X, n_variates, transformer)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joitran_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    n_variates = np.random.default_rng(42).normal(0, 1, 100)
    transformer = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_itransformer(X, n_variates, transformer)
    assert isinstance(result, dict)
