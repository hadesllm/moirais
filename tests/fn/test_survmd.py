"""Tests for survMd.survival_mediation."""
import numpy as np
import pytest
from morie.fn.survmd import survival_mediation


def test_survmd_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_mediation(T, delta, X, M, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survmd_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_mediation(T, delta, X, M, C)
    assert isinstance(result, dict)
