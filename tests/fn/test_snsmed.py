"""Tests for snsmed.sensitivity_mediation."""
import numpy as np
import pytest
from moirais.fn.snsmed import sensitivity_mediation


def test_snsmed_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    rho = 0.5
    result = sensitivity_mediation(X, M, Y, rho)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snsmed_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    rho = 0.5
    result = sensitivity_mediation(X, M, Y, rho)
    assert isinstance(result, dict)
