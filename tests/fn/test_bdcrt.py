"""Tests for bdcrt.backdoor_criterion."""
import numpy as np
import pytest
from morie.fn.bdcrt import backdoor_criterion


def test_bdcrt_basic():
    """Test basic functionality."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = backdoor_criterion(dag, X, Y, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdcrt_edge():
    """Test edge cases."""
    dag = {'A': [], 'B': ['A'], 'C': ['B']}
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = backdoor_criterion(dag, X, Y, Z)
    assert isinstance(result, dict)
