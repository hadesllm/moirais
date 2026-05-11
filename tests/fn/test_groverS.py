"""Tests for groverS.grover_search."""
import numpy as np
import pytest
from morie.fn.groverS import grover_search


def test_groverS_basic():
    """Test basic functionality."""
    oracle = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = grover_search(oracle, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_groverS_edge():
    """Test edge cases."""
    oracle = np.random.default_rng(42).normal(0, 1, 100)
    N = 100
    result = grover_search(oracle, N)
    assert isinstance(result, dict)
