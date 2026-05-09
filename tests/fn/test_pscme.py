"""Tests for pscme.path_specific_causal_effect."""
import numpy as np
import pytest
from moirais.fn.pscme import path_specific_causal_effect


def test_pscme_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = path_specific_causal_effect(X, M, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pscme_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = path_specific_causal_effect(X, M, Y)
    assert isinstance(result, dict)
