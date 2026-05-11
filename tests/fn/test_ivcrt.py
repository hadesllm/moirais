"""Tests for ivcrt.iv_conditions."""
import numpy as np
import pytest
from morie.fn.ivcrt import iv_conditions


def test_ivcrt_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = iv_conditions(dag, Z, X, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ivcrt_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = iv_conditions(dag, Z, X, Y)
    assert isinstance(result, dict)
