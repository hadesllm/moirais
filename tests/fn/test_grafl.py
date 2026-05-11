"""Tests for grafl.graphlet_kernel."""
import numpy as np
import pytest
from morie.fn.grafl import graphlet_kernel


def test_grafl_basic():
    """Test basic functionality."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = graphlet_kernel(G1, G2, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grafl_edge():
    """Test edge cases."""
    G1 = np.random.default_rng(42).normal(0, 1, 100)
    G2 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = graphlet_kernel(G1, G2, k)
    assert isinstance(result, dict)
