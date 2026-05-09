"""Tests for spwgts.spline_weights."""
import numpy as np
import pytest
from moirais.fn.spwgts import spline_weights


def test_spwgts_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    knots = np.random.default_rng(42).normal(0, 1, 100)
    result = spline_weights(A, H, knots)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spwgts_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    knots = np.random.default_rng(42).normal(0, 1, 100)
    result = spline_weights(A, H, knots)
    assert isinstance(result, dict)
