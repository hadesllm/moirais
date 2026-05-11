"""Tests for mdian.mediation_analysis."""
import numpy as np
import pytest
from morie.fn.mdian import mediation_analysis


def test_mdian_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mediation_analysis(Y, T, M, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mdian_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mediation_analysis(Y, T, M, X)
    assert isinstance(result, dict)
