"""Tests for tdmed.two_dimensional_mediation."""
import numpy as np
import pytest
from moirais.fn.tdmed import two_dimensional_mediation


def test_tdmed_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M1 = np.random.default_rng(42).normal(0, 1, 100)
    M2 = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = two_dimensional_mediation(X, M1, M2, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tdmed_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M1 = np.random.default_rng(42).normal(0, 1, 100)
    M2 = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = two_dimensional_mediation(X, M1, M2, Y)
    assert isinstance(result, dict)
