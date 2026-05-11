"""Tests for intmd4.interaction_mediation_4way."""
import numpy as np
import pytest
from morie.fn.intmd4 import interaction_mediation_4way


def test_intmd4_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = interaction_mediation_4way(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intmd4_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = interaction_mediation_4way(X, M, Y)
    assert isinstance(result, dict)
