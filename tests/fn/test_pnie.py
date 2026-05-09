"""Tests for pnie.pure_natural_indirect_effect."""
import numpy as np
import pytest
from moirais.fn.pnie import pure_natural_indirect_effect


def test_pnie_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = pure_natural_indirect_effect(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pnie_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = pure_natural_indirect_effect(X, M, Y)
    assert isinstance(result, dict)
