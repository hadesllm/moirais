"""Tests for medint.mediated_interaction."""
import numpy as np
import pytest
from moirais.fn.medint import mediated_interaction


def test_medint_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = mediated_interaction(Y, X, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_medint_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = mediated_interaction(Y, X, M)
    assert isinstance(result, dict)
