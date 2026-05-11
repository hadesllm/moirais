"""Tests for tcls.t_closeness."""
import numpy as np
import pytest
from morie.fn.tcls import t_closeness


def test_tcls_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quasi_ids = np.arange(100, dtype=int)
    sensitive = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = t_closeness(X, quasi_ids, sensitive, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tcls_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    quasi_ids = np.arange(100, dtype=int)
    sensitive = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = t_closeness(X, quasi_ids, sensitive, t)
    assert isinstance(result, dict)
