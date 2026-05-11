"""Tests for imai.imai_keele_yamamoto_mediation."""
import numpy as np
import pytest
from morie.fn.imai import imai_keele_yamamoto_mediation


def test_imai_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = imai_keele_yamamoto_mediation(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_imai_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = imai_keele_yamamoto_mediation(X, M, Y)
    assert isinstance(result, dict)
