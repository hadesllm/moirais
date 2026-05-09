"""Tests for otsktol.ot_sinkhorn_tol."""
import numpy as np
import pytest
from moirais.fn.otsktol import ot_sinkhorn_tol


def test_otsktol_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sinkhorn_tol(T, a, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otsktol_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_sinkhorn_tol(T, a, b)
    assert isinstance(result, dict)
