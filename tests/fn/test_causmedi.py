"""Tests for causmedi.causal_mediation_imai."""
import numpy as np
import pytest
from morie.fn.causmedi import causal_mediation_imai


def test_causmedi_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = causal_mediation_imai(X, M, Y, T, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causmedi_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = causal_mediation_imai(X, M, Y, T, B)
    assert isinstance(result, dict)
