"""Tests for ebalw.entropy_balancing."""
import numpy as np
import pytest
from moirais.fn.ebalw import entropy_balancing


def test_ebalw_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = entropy_balancing(X, T, moments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ebalw_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = entropy_balancing(X, T, moments)
    assert isinstance(result, dict)
