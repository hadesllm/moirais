"""Tests for binMd.binary_outcome_mediation."""
import numpy as np
import pytest
from moirais.fn.binMd import binary_outcome_mediation


def test_binMd_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = binary_outcome_mediation(Y, X, M, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_binMd_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = binary_outcome_mediation(Y, X, M, C)
    assert isinstance(result, dict)
