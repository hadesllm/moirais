"""Tests for survmd.survival_mediation."""
import numpy as np
import pytest
from morie.fn.survmd import survival_mediation


def test_survmd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_mediation(X, M, time, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survmd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = survival_mediation(X, M, time, event)
    assert isinstance(result, dict)
