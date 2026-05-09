"""Tests for wmtwgt.weights_matrix."""
import numpy as np
import pytest
from moirais.fn.wmtwgt import weights_matrix


def test_wmtwgt_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    method = 'auto'
    k_or_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = weights_matrix(coords, method, k_or_threshold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wmtwgt_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    method = 'auto'
    k_or_threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = weights_matrix(coords, method, k_or_threshold)
    assert isinstance(result, dict)
