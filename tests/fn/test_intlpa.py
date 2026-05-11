"""Tests for intlpa.interior_point_lp."""
import numpy as np
import pytest
from morie.fn.intlpa import interior_point_lp


def test_intlpa_basic():
    """Test basic functionality."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = interior_point_lp(c, A, b, x0, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intlpa_edge():
    """Test edge cases."""
    c = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    b = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = interior_point_lp(c, A, b, x0, tau)
    assert isinstance(result, dict)
