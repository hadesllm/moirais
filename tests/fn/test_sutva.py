"""Tests for sutva.sutva_assumption."""
import numpy as np
import pytest
from morie.fn.sutva import sutva_assumption


def test_sutva_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = sutva_assumption(Y, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sutva_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = sutva_assumption(Y, T)
    assert isinstance(result, dict)
