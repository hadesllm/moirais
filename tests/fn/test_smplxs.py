"""Tests for smplxs.simplex_lp."""
import numpy as np
import pytest
from moirais.fn.smplxs import simplex_lp


def test_smplxs_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = simplex_lp(c, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smplxs_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = simplex_lp(c, A, b)
    assert isinstance(result, dict)
