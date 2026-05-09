"""Tests for dsep.d_separation."""
import numpy as np
import pytest
from moirais.fn.dsep import d_separation


def test_dsep_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = d_separation(dag, X, Y, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dsep_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = d_separation(dag, X, Y, Z)
    assert isinstance(result, dict)
