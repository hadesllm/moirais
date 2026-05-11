"""Tests for agbckp.alphazero_backup."""
import numpy as np
import pytest
from morie.fn.agbckp import alphazero_backup


def test_agbckp_basic():
    """Test basic functionality."""
    leaf = np.random.default_rng(42).normal(0, 1, 100)
    value = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_backup(leaf, value, path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agbckp_edge():
    """Test edge cases."""
    leaf = np.random.default_rng(42).normal(0, 1, 100)
    value = np.random.default_rng(42).normal(0, 1, 100)
    path = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_backup(leaf, value, path)
    assert isinstance(result, dict)
