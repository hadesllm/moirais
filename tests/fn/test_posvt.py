"""Tests for posvt.positivity_assumption."""
import numpy as np
import pytest
from moirais.fn.posvt import positivity_assumption


def test_posvt_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = positivity_assumption(T, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_posvt_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = positivity_assumption(T, X)
    assert isinstance(result, dict)
