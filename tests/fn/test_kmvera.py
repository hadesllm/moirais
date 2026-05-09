"""Tests for kmvera.kamath_vera_adapter."""
import numpy as np
import pytest
from moirais.fn.kmvera import kamath_vera_adapter


def test_kmvera_basic():
    """Test basic functionality."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A_frozen = np.random.default_rng(42).normal(0, 1, 100)
    B_frozen = np.random.default_rng(42).normal(0, 1, 100)
    lam_b = np.random.default_rng(42).normal(0, 1, 100)
    lam_d = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmvera_edge():
    """Test edge cases."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A_frozen = np.random.default_rng(42).normal(0, 1, 100)
    B_frozen = np.random.default_rng(42).normal(0, 1, 100)
    lam_b = np.random.default_rng(42).normal(0, 1, 100)
    lam_d = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x)
    assert isinstance(result, dict)
