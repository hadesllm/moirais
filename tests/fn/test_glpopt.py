"""Tests for glpopt.glpk_lp."""
import numpy as np
import pytest
from moirais.fn.glpopt import glpk_lp


def test_glpopt_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = glpk_lp(c, A, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_glpopt_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = glpk_lp(c, A, b)
    assert isinstance(result, dict)
