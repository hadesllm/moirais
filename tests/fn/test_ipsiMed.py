"""Tests for ipsiMed.interventional_psi."""
import numpy as np
import pytest
from moirais.fn.ipsiMed import interventional_psi


def test_ipsiMed_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = interventional_psi(Y, X, M, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ipsiMed_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = interventional_psi(Y, X, M, C)
    assert isinstance(result, dict)
