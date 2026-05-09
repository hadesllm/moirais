"""Tests for itrct1.interaction_did."""
import numpy as np
import pytest
from moirais.fn.itrct1 import interaction_did


def test_itrct1_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = interaction_did(y, D, V, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_itrct1_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = interaction_did(y, D, V, X)
    assert isinstance(result, dict)
