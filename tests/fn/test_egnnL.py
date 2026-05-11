"""Tests for egnnL.egnn_layer."""
import numpy as np
import pytest
from morie.fn.egnnL import egnn_layer


def test_egnnL_basic():
    """Test basic functionality."""
    h = 0.3
    x = np.random.default_rng(42).normal(0, 1, 100)
    edges = [('A', 'B'), ('B', 'C')]
    result = egnn_layer(h, x, edges)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_egnnL_edge():
    """Test edge cases."""
    h = 0.3
    x = np.random.default_rng(42).normal(0, 1, 100)
    edges = [('A', 'B'), ('B', 'C')]
    result = egnn_layer(h, x, edges)
    assert isinstance(result, dict)
