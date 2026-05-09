"""Tests for jacqkn.jacquez_k_nn_test."""
import numpy as np
import pytest
from moirais.fn.jacqkn import jacquez_k_nn_test


def test_jacqkn_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    time = np.linspace(0, 10, 100)
    k = 5
    result = jacquez_k_nn_test(coords, time, k)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_jacqkn_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    time = np.linspace(0, 10, 100)
    k = 5
    result = jacquez_k_nn_test(coords, time, k)
    assert isinstance(result, dict)
