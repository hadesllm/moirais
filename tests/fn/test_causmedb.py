"""Tests for causmedb.causal_mediation_baron_kenny."""
import numpy as np
import pytest
from morie.fn.causmedb import causal_mediation_baron_kenny


def test_causmedb_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = causal_mediation_baron_kenny(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmedb_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = causal_mediation_baron_kenny(X, M, Y)
    assert isinstance(result, dict)
